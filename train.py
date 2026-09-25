"""Train / evaluate IRENE-MM with patient-grouped cross-validation.

Engineering features
  * PyTorch DataLoader with worker processes, pinned memory, persistent workers, prefetching
  * Automatic mixed precision: fp16 + GradScaler on CUDA, bf16 autocast on CPU (--amp)
  * DistributedDataParallel via torchrun (NCCL on GPU, Gloo on CPU); DistributedSampler
  * Weights & Biases experiment tracking (offline by default; `wandb sync` to upload)

Example
    python train.py --name irene --fusion_layer 2 --cross_mode bi_avg --seeds 0 1 2
    torchrun --nproc_per_node 2 train.py --name ddp_check --folds 0 --epochs 3
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from models.irene_mm import IRENEMM, MMConfig

LABELS = ["COVID-19", "Viral (other)", "Bacterial", "Fungal", "Pneumonia (other/unspec.)", "Tuberculosis", "No Finding"]


class CXRMultimodal(Dataset):
    """Image tokens + note tokens + demographics + structured values + multi-hot label."""

    def __init__(self, store, idx, train, text_field="masked", doc=None):
        self.s, self.idx, self.train, self.doc = store, np.asarray(idx), train, doc
        self.txt = store["txt_" + text_field]
        self.len = store["len_" + text_field]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        s = self.s
        v = np.random.randint(s["img"].shape[1]) if self.train else 0   # cached augmented views
        return {
            "img": torch.from_numpy(s["img"][i, v].astype(np.float32)),
            "txt": torch.from_numpy(self.txt[i].astype(np.float32)),
            "txt_len": torch.tensor(int(self.len[i])),
            "demo": torch.from_numpy(s["demo"][i]),
            "lab": torch.from_numpy(s["lab"][i]),
            "lab_mask": torch.from_numpy(s["lab_mask"][i]),
            "y": torch.from_numpy(s["y"][i]),
            "text_id": torch.tensor(int(s["text_id"][i])),
            "doc": torch.from_numpy(self.doc[i]) if self.doc is not None else torch.zeros(1),
            "idx": torch.tensor(int(i)),
        }


def load_store(processed, img_grid=7, max_txt=64):
    with open(os.path.join(processed, "covidcxr_mm.pkl"), "rb") as fh:
        samples = pickle.load(fh)
    f = np.load(os.path.join(processed, "features.npz"))
    keys = list(f["keys"])
    assert keys == list(samples)
    st = {k: f[k] for k in ["img", "txt_raw", "len_raw", "txt_masked", "len_masked"]}
    if img_grid != 7:   # pool the 7x7 CNN grid to img_grid x img_grid tokens (compute budget)
        a = torch.from_numpy(st["img"]).float()                  # [N,V,49,C]
        N, V, _, C = a.shape
        a = a.view(N * V, 7, 7, C).permute(0, 3, 1, 2)
        a = F.adaptive_avg_pool2d(a, img_grid).permute(0, 2, 3, 1).reshape(N, V, img_grid ** 2, C)
        st["img"] = a.half().numpy()
    for fld in ["raw", "masked"]:
        st["txt_" + fld] = st["txt_" + fld][:, :max_txt]
        st["len_" + fld] = np.minimum(st["len_" + fld], max_txt)
    st["demo"] = np.stack([samples[k]["bics"] for k in keys]).astype(np.float32)
    st["lab"] = np.stack([samples[k]["bts"] for k in keys]).astype(np.float32)
    st["lab_mask"] = np.stack([samples[k]["bts_mask"] for k in keys]).astype(np.float32)
    st["y"] = np.stack([samples[k]["label"] for k in keys]).astype(np.float32)
    st["fold"] = np.array([samples[k]["fold"] for k in keys])
    st["patient"] = np.array([samples[k]["patient"] for k in keys])
    st["text_id"] = np.array([int(hashlib.md5(samples[k]["text_raw"].encode()).hexdigest()[:8], 16)
                              if samples[k]["text_raw"] else -1 - i for i, k in enumerate(keys)])
    st["keys"] = keys
    st["text_raw"] = np.array([samples[k]["text_raw"] for k in keys], dtype=object)
    st["text_masked"] = np.array([samples[k]["text_masked"] for k in keys], dtype=object)
    return st


def metrics(y, p):
    aucs = [roc_auc_score(y[:, c], p[:, c]) if 0 < y[:, c].sum() < len(y) else np.nan for c in range(y.shape[1])]
    aps = [average_precision_score(y[:, c], p[:, c]) if y[:, c].sum() > 0 else np.nan for c in range(y.shape[1])]
    pred = np.zeros_like(p)
    pred[np.arange(len(p)), p.argmax(1)] = 1
    return {"macro_auroc": float(np.nanmean(aucs)), "macro_auprc": float(np.nanmean(aps)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float((p.argmax(1) == y.argmax(1)).mean()),
            "per_class_auroc": {l: float(a) for l, a in zip(LABELS, aucs)}}


def retrieval(ei, et, tid, has):
    """Image->text retrieval over unique notes in the evaluation set (R@1/5/10, median rank)."""
    ei, et, tid = ei[has], et[has], tid[has]
    uniq, first = np.unique(tid, return_index=True)
    T = et[first]
    sim = ei @ T.T
    gt = np.searchsorted(uniq, tid)
    ranks = (sim > sim[np.arange(len(sim)), gt][:, None]).sum(1) + 1
    return {"R@1": float((ranks <= 1).mean()), "R@5": float((ranks <= 5).mean()),
            "R@10": float((ranks <= 10).mean()), "median_rank": float(np.median(ranks)),
            "n_queries": int(len(ranks)), "n_gallery": int(len(uniq))}


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def run_model(model, b, drop_text=None):
    return model(b["img"], b["txt"], b["txt_len"], b["demo"], b["lab"], b["lab_mask"], drop_text=drop_text,
                 doc=b["doc"])


def doc_features(texts, fit_idx, dim, seed):
    """Document-level note embedding: TF-IDF (1-2 grams) -> truncated SVD, fitted on the training
    patients of the current fold only (no test-fold information), then applied to every note.
    Plays the role of a sentence-level [CLS] embedding such as the BERT features used by IRENE."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xf = vec.fit_transform(texts[fit_idx])
    svd = TruncatedSVD(n_components=dim, random_state=seed).fit(Xf)
    Z = svd.transform(vec.transform(texts)).astype(np.float32)
    Z = Z / (Z[fit_idx].std(0, keepdims=True) + 1e-6)
    Z[np.array([not t for t in texts])] = 0.0
    return Z


@torch.no_grad()
def predict(model, loader, dev, amp_dtype, drop_text_all=False):
    model.eval()
    P, E_i, E_t, H, I = [], [], [], [], []
    for b in loader:
        b = to_dev(b, dev)
        dt = torch.ones(len(b["y"]), dtype=torch.bool, device=dev) if drop_text_all else None
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            o = run_model(model, b, dt)
        P.append(torch.sigmoid(o["logits"].float()).cpu())
        I.append(b["idx"].cpu())
        if "emb_i" in o:
            E_i.append(o["emb_i"].float().cpu()); E_t.append(o["emb_t"].float().cpu()); H.append(o["has_txt"].cpu())
    out = {"p": torch.cat(P).numpy(), "idx": torch.cat(I).numpy()}
    if E_i:
        out.update(ei=torch.cat(E_i).numpy(), et=torch.cat(E_t).numpy(), has=torch.cat(H).numpy())
    return out


def train_fold(args, st, cfg, fold, seed, dev, world, rank, wb):
    torch.manual_seed(seed); np.random.seed(seed)
    test_idx = np.where(st["fold"] == fold)[0]
    trval = np.where(st["fold"] != fold)[0]
    # patient-grouped inner validation split for early stopping (test fold never touched)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_rel, va_rel = next(gss.split(trval, groups=st["patient"][trval]))
    tr_idx, va_idx = trval[tr_rel], trval[va_rel]

    doc = None
    if cfg.doc_dim > 0:
        doc = doc_features(st["text_" + args.text_field], tr_idx, cfg.doc_dim, seed)
    ds_tr = CXRMultimodal(st, tr_idx, True, args.text_field, doc)
    sampler = DistributedSampler(ds_tr, world, rank, shuffle=True, seed=seed) if world > 1 else None
    kw = dict(num_workers=args.workers, pin_memory=dev.type == "cuda",
              persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(CXRMultimodal(st, va_idx, False, args.text_field, doc), batch_size=128, **kw)
    dl_te = DataLoader(CXRMultimodal(st, test_idx, False, args.text_field, doc), batch_size=128, **kw)

    model = IRENEMM(cfg).to(dev)
    if world > 1:
        model = DDP(model, device_ids=[dev.index] if dev.type == "cuda" else None)
    core = model.module if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    amp_dtype = (torch.float16 if dev.type == "cuda" else torch.bfloat16) if args.amp else None
    scaler = torch.amp.GradScaler(enabled=args.amp and dev.type == "cuda")

    pos_w = None
    if args.pos_weight == "sqrt":
        yp = st["y"][tr_idx].sum(0)
        pos_w = torch.tensor(np.sqrt((len(tr_idx) - yp) / np.maximum(yp, 1)), dtype=torch.float32, device=dev)
    best, best_state, bad = -np.inf, None, 0
    hist = []
    for ep in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        t0, tl, tc, n = time.time(), 0.0, 0.0, 0
        for b in dl_tr:
            b = to_dev(b, dev)
            dt = None
            if cfg.modality_dropout > 0:
                dt = torch.rand(len(b["y"]), device=dev) < cfg.modality_dropout
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                o = run_model(model, b, dt)
                loss_cls = F.binary_cross_entropy_with_logits(o["logits"].float(), b["y"], pos_weight=pos_w)
                loss_con = core.contrastive_loss(o, b["text_id"]) if cfg.contrastive != "none" else torch.zeros((), device=dev)
                loss = loss_cls + args.lam * loss_con
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss_cls.item() * len(b["y"]); tc += loss_con.item() * len(b["y"]); n += len(b["y"])
        pv = predict(core, dl_va, dev, amp_dtype)
        yv = st["y"][pv["idx"]]
        va_auc = metrics(yv, pv["p"])["macro_auroc"]
        pc = np.clip(pv["p"], 1e-6, 1 - 1e-6)
        va_bce = float(-(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)).mean())
        va = -va_bce if args.select == "bce" else va_auc
        rec = {"epoch": ep, "train_bce": tl / n, "train_con": tc / n, "val_auroc": va_auc, "val_bce": va_bce,
               "sec": time.time() - t0, "samples_per_sec": n * world / (time.time() - t0)}
        hist.append(rec)
        if wb is not None and rank == 0:
            wb.log({f"fold{fold}/{k}": v for k, v in rec.items()})
        if va > best + 1e-4:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in core.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    core.load_state_dict(best_state)
    pt = predict(core, dl_te, dev, amp_dtype)
    res = {"test": pt, "hist": hist, "best_val": float(best)}
    if args.eval_missing_text:
        res["test_notext"] = predict(core, dl_te, dev, amp_dtype, drop_text_all=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.0, help="weight of the contrastive loss")
    ap.add_argument("--text_field", default="masked", choices=["masked", "raw"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--wandb", default="offline", choices=["offline", "online", "disabled"])
    ap.add_argument("--eval_missing_text", action="store_true")
    ap.add_argument("--pos_weight", default="sqrt", choices=["none", "sqrt"])
    ap.add_argument("--select", default="bce", choices=["bce", "auroc"], help="validation criterion for early stopping")
    ap.add_argument("--img_grid", type=int, default=4, help="image token grid (7 = full 7x7 DenseNet grid)")
    ap.add_argument("--max_txt", type=int, default=48, help="max note tokens")
    for f, v in asdict(MMConfig()).items():
        if f in ("img_in", "n_img_tok", "txt_in", "max_txt", "n_lab", "num_classes", "pixel_input", "patch"):
            continue
        t = (lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else type(v)
        ap.add_argument(f"--{f}", type=t, default=v)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.threads)

    st = load_store(args.processed, args.img_grid, args.max_txt)
    cfg = MMConfig(**{k: getattr(args, k) for k in asdict(MMConfig()) if hasattr(args, k) and k != "max_txt"},
                   max_txt=st["txt_masked"].shape[1], n_img_tok=st["img"].shape[2], n_lab=st["lab"].shape[1], num_classes=st["y"].shape[1])
    if not cfg.use_img or not (cfg.use_txt or cfg.use_tab):
        cfg.fusion_layer = 0
    if cfg.contrastive == "none":
        args.lam = 0.0

    wb = None
    if args.wandb != "disabled" and rank == 0:
        import wandb
        os.environ.setdefault("WANDB_SILENT", "true")
        wb = wandb.init(project="irene-mm-covidcxr", name=args.name, mode=args.wandb,
                        config={**vars(args), **asdict(cfg)}, dir="results", reinit=True)

    os.makedirs(args.out, exist_ok=True)
    summary = {"name": args.name, "config": asdict(cfg), "args": vars(args), "seeds": {}}
    n_params = sum(p.numel() for p in IRENEMM(cfg).parameters())
    summary["n_params"] = n_params
    t_start = time.time()
    for seed in args.seeds:
        oof = np.full(st["y"].shape, np.nan, dtype=np.float32)
        oof_nt = np.full(st["y"].shape, np.nan, dtype=np.float32)
        emb = {}
        fold_res, hists = {}, {}
        for fold in args.folds:
            r = train_fold(args, st, cfg, fold, seed, dev, world, rank, wb)
            te = r["test"]
            oof[te["idx"]] = te["p"]
            m = metrics(st["y"][te["idx"]], te["p"])
            if "ei" in te:
                m["retrieval"] = retrieval(te["ei"], te["et"], st["text_id"][te["idx"]], te["has"])
            if "test_notext" in r:
                oof_nt[r["test_notext"]["idx"]] = r["test_notext"]["p"]
            fold_res[fold] = m
            hists[fold] = r["hist"]
            if rank == 0:
                print(f"[{args.name}] seed {seed} fold {fold}: AUROC {m['macro_auroc']:.4f} "
                      f"acc {m['accuracy']:.3f} epochs {len(r['hist'])}", flush=True)
                if wb is not None:
                    wb.log({f"seed{seed}/fold{fold}/test_macro_auroc": m["macro_auroc"]})
        done = ~np.isnan(oof[:, 0])
        pooled = metrics(st["y"][done], oof[done])
        pooled["fold_mean_auroc"] = float(np.mean([v["macro_auroc"] for v in fold_res.values()]))
        if any("retrieval" in v for v in fold_res.values()):
            pooled["retrieval"] = {k: float(np.mean([v["retrieval"][k] for v in fold_res.values()]))
                                   for k in ["R@1", "R@5", "R@10", "median_rank"]}
        if args.eval_missing_text:
            pooled["notext"] = metrics(st["y"][done], oof_nt[done])
        summary["seeds"][seed] = {"pooled": pooled, "folds": fold_res, "hist": hists}
        np.save(os.path.join(args.out, f"{args.name}_seed{seed}_oof.npy"), oof)
        if rank == 0:
            print(f"[{args.name}] seed {seed} pooled OOF macro AUROC {pooled['macro_auroc']:.4f}", flush=True)
    summary["wall_sec"] = time.time() - t_start
    aucs = [v["pooled"]["macro_auroc"] for v in summary["seeds"].values()]
    summary["macro_auroc_mean"], summary["macro_auroc_std"] = float(np.mean(aucs)), float(np.std(aucs))
    if rank == 0:
        with open(os.path.join(args.out, f"{args.name}.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        if wb is not None:
            wb.summary.update({"macro_auroc_mean": summary["macro_auroc_mean"],
                               "macro_auroc_std": summary["macro_auroc_std"], "n_params": n_params})
            wb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
