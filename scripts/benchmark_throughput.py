"""Data-loading and training-throughput benchmarks for the multimodal pipeline.

1. Input pipeline: on-the-fly decode (PIL decode + resize per sample) vs a pre-resized uint8
   memory-mapped cache, across DataLoader settings (workers, pin_memory, persistent workers).
2. Training step: end-to-end pixel-input IRENE-MM, fp32 vs AMP (bf16 autocast on CPU, fp16 on GPU).
3. DDP: run separately with torchrun (see --ddp flag) -- reports per-rank and global samples/sec.

Results are written to results/throughput.json. All numbers are measured on the machine the
script runs on; the paper reports the hardware it was run on.
"""
import argparse
import json
import os
import pickle
import platform
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.irene_mm import IRENEMM, MMConfig  # noqa: E402

RAW = "data/raw/covid-chestxray-dataset"
PROC = "data/processed"


def load_samples():
    with open(os.path.join(PROC, "covidcxr_mm.pkl"), "rb") as fh:
        return pickle.load(fh)


class DecodeDS(Dataset):
    """Baseline: decode the original JPEG/PNG (often >2000px) and resize for every sample."""

    def __init__(self, samples, feats):
        self.k = list(samples)
        self.s = samples
        self.f = feats
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def __len__(self):
        return len(self.k)

    def __getitem__(self, i):
        img = self.tf(Image.open(os.path.join(RAW, self.s[self.k[i]]["image"])).convert("L"))
        return img, torch.from_numpy(self.f["txt_masked"][i].astype(np.float32)), int(self.f["len_masked"][i])


class CacheDS(Dataset):
    """Optimised: images pre-resized once to 256x256 uint8 in a memory-mapped array (shard-friendly);
    per-sample work is a random crop + dtype conversion."""

    def __init__(self, cache, feats, train=True):
        self.m = np.load(cache, mmap_mode="r")
        self.f = feats
        self.train = train

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        a = self.m[i]
        if self.train:
            y, x = np.random.randint(0, 33, 2)
        else:
            y = x = 16
        img = torch.from_numpy(np.ascontiguousarray(a[y:y + 224, x:x + 224])).float().div_(255)[None]
        return img, torch.from_numpy(self.f["txt_masked"][i].astype(np.float32)), int(self.f["len_masked"][i])


def build_cache(samples, path):
    if os.path.exists(path):
        return
    tf = T.Compose([T.Resize(256), T.CenterCrop(256)])
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(samples), 256, 256))
    for i, k in enumerate(samples):
        arr[i] = np.asarray(tf(Image.open(os.path.join(RAW, samples[k]["image"])).convert("L")))
    arr.flush()


def time_loader(ds, workers, pin, persistent, bs=32, epochs=2):
    kw = dict(batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
    if workers > 0:
        kw.update(persistent_workers=persistent, prefetch_factor=4)
    dl = DataLoader(ds, **kw)
    n, t0 = 0, time.time()
    for _ in range(epochs):
        for b in dl:
            n += len(b[0])
    return n / (time.time() - t0)


def pixel_batch(bs, feats, dev):
    return {"img": torch.rand(bs, 1, 224, 224, device=dev),
            "txt": torch.from_numpy(feats["txt_masked"][:bs].astype(np.float32)).to(dev),
            "txt_len": torch.from_numpy(feats["len_masked"][:bs].astype(np.int64)).to(dev),
            "demo": torch.zeros(bs, 4, device=dev), "lab": torch.zeros(bs, 11, device=dev),
            "lab_mask": torch.zeros(bs, 11, device=dev), "y": torch.zeros(bs, 7, device=dev)}


def time_train(amp, dev, feats, bs=16, steps=12, cfg=None):
    cfg = cfg or MMConfig(pixel_input=True, hidden=256, layers=6, heads=4, mlp=512)
    model = IRENEMM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    dtype = (torch.float16 if dev.type == "cuda" else torch.bfloat16)
    scaler = torch.amp.GradScaler(enabled=amp and dev.type == "cuda")
    b = pixel_batch(bs, feats, dev)
    ts = []
    for s in range(steps):
        t0 = time.time()
        with torch.autocast(dev.type, dtype=dtype, enabled=amp):
            o = model(b["img"], b["txt"], b["txt_len"], b["demo"], b["lab"], b["lab_mask"])
            loss = F.binary_cross_entropy_with_logits(o["logits"].float(), b["y"])
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if s >= 2:
            ts.append(time.time() - t0)
    return bs / np.mean(ts)


def ddp_bench(feats, steps=12, bs=16):
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() // world))
    model = torch.nn.parallel.DistributedDataParallel(
        IRENEMM(MMConfig(pixel_input=True, hidden=256, layers=6, heads=4, mlp=512)).to(dev))
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    b = pixel_batch(bs, feats, dev)
    ts = []
    for s in range(steps):
        t0 = time.time()
        o = model(b["img"], b["txt"], b["txt_len"], b["demo"], b["lab"], b["lab_mask"])
        loss = F.binary_cross_entropy_with_logits(o["logits"], b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
        if s >= 2:
            ts.append(time.time() - t0)
    sps = torch.tensor(bs / np.mean(ts))
    dist.all_reduce(sps)
    if rank == 0:
        path = "results/throughput.json"
        res = json.load(open(path)) if os.path.exists(path) else {}
        res[f"ddp_world{world}_global_samples_per_sec"] = float(sps)
        res[f"ddp_world{world}_threads_per_rank"] = torch.get_num_threads()
        json.dump(res, open(path, "w"), indent=1)
        print("DDP", world, float(sps))
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()
    feats = dict(np.load(os.path.join(PROC, "features.npz")))
    feats = {k: feats[k] for k in ["txt_masked", "len_masked"]}
    if args.ddp:
        return ddp_bench(feats)
    samples = load_samples()
    os.makedirs("results", exist_ok=True)
    cache = os.path.join(PROC, "img256_uint8.npy")
    t0 = time.time()
    build_cache(samples, cache)
    res = {"host": {"cpu_count": os.cpu_count(), "cuda": torch.cuda.is_available(),
                    "torch": torch.__version__, "platform": platform.platform()},
           "cache_build_sec": time.time() - t0, "loader": []}
    for name, ds in [("decode", DecodeDS(samples, feats)), ("mmap_cache", CacheDS(cache, feats))]:
        for workers in [0, 1, 2]:
            for pin in [False, True]:
                for persistent in ([False, True] if workers > 0 else [False]):
                    sps = time_loader(ds, workers, pin, persistent, epochs=1 if name == "decode" else 3)
                    res["loader"].append({"dataset": name, "workers": workers, "pin_memory": pin,
                                          "persistent": persistent, "samples_per_sec": sps})
                    print(res["loader"][-1], flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(os.cpu_count())
    res["train_fp32_samples_per_sec"] = time_train(False, dev, feats)
    res["train_amp_samples_per_sec"] = time_train(True, dev, feats)
    res["amp_dtype"] = "float16" if dev.type == "cuda" else "bfloat16"
    old = json.load(open("results/throughput.json")) if os.path.exists("results/throughput.json") else {}
    old.update(res)
    json.dump(old, open("results/throughput.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "loader"}, indent=1))


if __name__ == "__main__":
    main()
