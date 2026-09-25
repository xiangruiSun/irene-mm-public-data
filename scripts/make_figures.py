"""Aggregate experiment results into paper tables (LaTeX) and figures (PDF).

    python scripts/make_figures.py
Outputs: paper/figures/*.pdf, paper/tables/*.tex, results/summary.json
"""
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

RUNS = "results/runs"
FIG = "paper/figures"
TAB = "paper/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# reference categorical palette (validated order) + text tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
LABELS = ["COVID-19", "Viral (other)", "Bacterial", "Fungal", "Pneumonia (other/unspec.)", "Tuberculosis", "No Finding"]


def load_runs():
    R = {}
    for f in glob.glob(os.path.join(RUNS, "*.json")):
        d = json.load(open(f))
        R[d["name"]] = d
    return R


def seed_vals(d, key="macro_auroc", sub=None):
    out = []
    for s in d["seeds"].values():
        p = s["pooled"] if sub is None else s["pooled"][sub]
        out.append(p[key])
    return np.array(out)


def ms(d, key="macro_auroc", sub=None):
    v = seed_vals(d, key, sub)
    return v.mean(), v.std()


def fmt(m, s, bold=False):
    t = f"{m:.3f}$\\pm${s:.3f}"
    return f"\\textbf{{{t}}}" if bold else t


def store():
    samples = pickle.load(open("data/processed/covidcxr_mm.pkl", "rb"))
    keys = list(samples)
    y = np.stack([samples[k]["label"] for k in keys])
    pat = np.array([samples[k]["patient"] for k in keys])
    return samples, keys, y, pat


def mixed_mask(samples):
    src = np.array([s["source"] for s in samples.values()])
    return np.isin(src, ["radiopaedia.org", "eurorad.org"])


def seed_oofs(name):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(RUNS, f"{name}_seed*_oof.npy")))]


def mean_oof(name):
    fs = sorted(glob.glob(os.path.join(RUNS, f"{name}_seed*_oof.npy")))
    return np.mean([np.load(f) for f in fs], 0)


def macro_auc(y, p):
    return np.nanmean([roc_auc_score(y[:, c], p[:, c]) if 0 < y[:, c].sum() < len(y) else np.nan
                       for c in range(y.shape[1])])


def paired_bootstrap(y, pat, pa, pb, B=1000, seed=0):
    """Patient-level paired bootstrap of macro-AUROC difference (a - b)."""
    rng = np.random.default_rng(seed)
    up = np.unique(pat)
    idx_by_p = {p: np.where(pat == p)[0] for p in up}
    diffs = []
    for _ in range(B):
        ps = rng.choice(up, len(up), replace=True)
        ii = np.concatenate([idx_by_p[p] for p in ps])
        diffs.append(macro_auc(y[ii], pa[ii]) - macro_auc(y[ii], pb[ii]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(macro_auc(y, pa) - macro_auc(y, pb)), float(lo), float(hi), float(max(p, 1 / B))


def boot_ci(y, pat, p, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    up = np.unique(pat)
    idx_by_p = {q: np.where(pat == q)[0] for q in up}
    v = []
    for _ in range(B):
        ii = np.concatenate([idx_by_p[q] for q in rng.choice(up, len(up), replace=True)])
        v.append(macro_auc(y[ii], p[ii]))
    return np.percentile(v, [2.5, 97.5])


# ------------------------------------------------------------------ figures
def fig_dataset(samples, y):
    st = json.load(open("data/processed/stats.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    cnt = y.sum(0).astype(int)
    order = np.argsort(cnt)
    ax[0].barh(np.arange(7), cnt[order], color=C[0], height=0.62)
    ax[0].set_yticks(np.arange(7), [LABELS[i].replace(" (other/unspec.)", " (other)") for i in order])
    for i, v in enumerate(cnt[order]):
        ax[0].text(v + 8, i, str(v), va="center", fontsize=7, color=INK2)
    ax[0].set_xscale("log"); ax[0].set_xlim(8, 1500); ax[0].set_xlabel("images (log scale)")
    ax[0].set_title("(a) Diagnosis labels", loc="left"); ax[0].grid(axis="y", visible=False)

    words = [len(s["text_raw"].split()) for s in samples.values() if s["text_raw"]]
    ax[1].hist(np.clip(words, 0, 400), bins=np.arange(0, 410, 20), color=C[0], edgecolor="white", linewidth=0.6)
    ax[1].axvline(np.median(words), color=INK, lw=1, ls="--")
    ax[1].text(np.median(words) + 8, ax[1].get_ylim()[1] * 0.85, f"median {int(np.median(words))}", fontsize=7, color=INK)
    ax[1].set_xlabel("words per note (clipped at 400)"); ax[1].set_ylabel("notes")
    ax[1].set_title("(b) Unstructured text", loc="left")

    mr = st["missing_rate"]
    names = {"sex": "sex", "age": "age", "offset": "days since onset", "temperature": "temperature",
             "pO2_saturation": "SpO$_2$", "leukocyte_count": "leukocytes", "neutrophil_count": "neutrophils",
             "lymphocyte_count": "lymphocytes"}
    ks = sorted(mr, key=lambda k: mr[k])
    obs = [100 * (1 - mr[k]) for k in ks]
    ax[2].barh(np.arange(len(ks)), obs, color=C[0], height=0.62)
    ax[2].set_yticks(np.arange(len(ks)), [names[k] for k in ks]); ax[2].invert_yaxis()
    ax[2].set_xlim(0, 100); ax[2].set_xlabel("% of images observed")
    for i, v in enumerate(obs):
        ax[2].text(v + 2, i, f"{v:.0f}%", va="center", fontsize=7, color=INK2)
    ax[2].set_title("(c) Structured fields", loc="left"); ax[2].grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(f"{FIG}/dataset.pdf"); plt.close(fig)


def bar_panel(ax, names, labels, R, colors, highlight=None, key="macro_auroc", sub=None, xlim=None):
    m = np.array([ms(R[n], key, sub)[0] for n in names])
    s = np.array([ms(R[n], key, sub)[1] for n in names])
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, m, xerr=s, color=colors, height=0.62, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK2})
    ax.set_yticks(ypos, labels)
    for yy, mm, ss in zip(ypos, m, s):
        ax.text(mm + ss + 0.004, yy, f"{mm:.3f}", va="center", fontsize=7, color=INK)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def fig_main(R):
    names = ["img_only", "tab_only", "txt_only", "txt_tab_only", "img_tab", "early_fusion", "late_fusion", "irene_k2"]
    labels = ["Image only", "Structured only", "Note only", "Note + structured", "Image + structured",
              "Early fusion (k=0)", "Late fusion (k=6)", "IRENE mid fusion (k=2)"]
    names = [n for n in names if n in R]
    labels = labels[-len(names):] if len(names) < 8 else labels
    cols = [C[0] if n.endswith("only") or n == "img_tab" else C[1] for n in names]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bar_panel(ax, names, labels, R, cols, xlim=(0.5, 0.82))
    ax.set_xlabel("macro AUROC (pooled OOF, mean$\\pm$sd)")
    fig.savefig(f"{FIG}/main_results.pdf"); plt.close(fig)


def fig_fusion(R):
    ks = [k for k in range(7) if (f"irene_k{k}" in R) or (k == 0 and "early_fusion" in R)]
    get = lambda k: R["early_fusion"] if k == 0 else R[f"irene_k{k}"]
    m = np.array([ms(get(k))[0] for k in ks]); s = np.array([ms(get(k))[1] for k in ks])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax[0].fill_between(ks, m - s, m + s, color=C[0], alpha=0.15, lw=0)
    ax[0].plot(ks, m, "-o", color=C[0], lw=2, ms=5, label="IRENE bidirectional attention")
    if "late_fusion" in R:
        lm, ls = ms(R["late_fusion"])
        ax[0].errorbar([6], [lm], yerr=[ls], fmt="s", color=C[1], ms=5, capsize=2, label="late fusion, no cross-attn")
    if "img_only" in R:
        im, _ = ms(R["img_only"])
        ax[0].axhline(im, color=INK2, lw=1, ls=":", label="image only")
    ax[0].set_xticks(range(7), ["0\nearly", "1", "2\nIRENE", "3", "4", "5", "6\nlate"])
    ax[0].set_xlabel("fusion position k (dual-stream blocks before joint self-attention)")
    ax[0].set_ylabel("macro AUROC"); ax[0].set_title("(a) Where to fuse", loc="left")
    ax[0].legend(fontsize=7, loc="lower center")
    # retrieval-free companion: macro AUPRC
    m2 = np.array([ms(get(k), "macro_auprc")[0] for k in ks]); s2 = np.array([ms(get(k), "macro_auprc")[1] for k in ks])
    ax[1].fill_between(ks, m2 - s2, m2 + s2, color=C[0], alpha=0.15, lw=0)
    ax[1].plot(ks, m2, "-o", color=C[0], lw=2, ms=5)
    if "late_fusion" in R:
        lm, ls = ms(R["late_fusion"], "macro_auprc")
        ax[1].errorbar([6], [lm], yerr=[ls], fmt="s", color=C[1], ms=5, capsize=2)
    ax[1].set_xticks(range(7), ["0\nearly", "1", "2\nIRENE", "3", "4", "5", "6\nlate"])
    ax[1].set_xlabel("fusion position k"); ax[1].set_ylabel("macro AUPRC"); ax[1].set_title("(b) Same sweep, macro AUPRC", loc="left")
    fig.tight_layout(w_pad=2)
    fig.savefig(f"{FIG}/fusion_position.pdf"); plt.close(fig)


def fig_attn(R):
    names = ["attn_none", "attn_i2t", "attn_t2i", "attn_bi_cross", "attn_gated", "irene_k2"]
    labels = ["No cross-attn (self only)", "Image$\\to$text only", "Text$\\to$image only",
              "Co-attention (cross only)", "Gated cross-attn (tanh)", "IRENE (self+cross)/2"]
    keep = [i for i, n in enumerate(names) if n in R]
    names, labels = [names[i] for i in keep], [labels[i] for i in keep]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
    bar_panel(ax[0], names, labels, R, [C[1] if n == "irene_k2" else C[0] for n in names], xlim=(0.6, 0.85))
    ax[0].set_xlabel("macro AUROC"); ax[0].set_title("(a) Full input", loc="left")
    bar_panel(ax[1], names, labels, R, [C[1] if n == "irene_k2" else C[0] for n in names], sub="notext", xlim=(0.5, 0.85))
    ax[1].set_xlabel("macro AUROC, note removed"); ax[1].set_title("(b) Missing-note robustness", loc="left")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(f"{FIG}/attention_structure.pdf"); plt.close(fig)


def fig_contrastive(R):
    lams = ["0.1", "0.5", "1.0"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    base = ms(R["irene_k2"]) if "irene_k2" in R else None
    for j, (obj, lab) in enumerate([("clip", "CLIP / InfoNCE"), ("siglip", "SigLIP (sigmoid)")]):
        ns = [f"{obj}_l{l}" for l in lams if f"{obj}_l{l}" in R]
        if not ns:
            continue
        x = [0] + [float(n.split("_l")[1]) for n in ns]
        m = [base[0]] + [ms(R[n])[0] for n in ns]
        s = [base[1]] + [ms(R[n])[1] for n in ns]
        ax[0].errorbar(x, m, yerr=s, fmt="-o", color=C[j], lw=2, ms=5, capsize=2, label=lab)
        r5 = [ms(R[n], "R@5", "retrieval")[0] * 100 for n in ns]
        r5s = [ms(R[n], "R@5", "retrieval")[1] * 100 for n in ns]
        ax[1].errorbar(x[1:], r5, yerr=r5s, fmt="-o", color=C[j], lw=2, ms=5, capsize=2, label=lab)
    for n, mk, lab, col in [("late_clip_l0.5", "s", "CLIP, late fusion (dual encoder)", C[2]),
                            ("early_clip_l0.5", "^", "CLIP, early fusion (token level)", C[4])]:
        if n in R:
            ax[0].errorbar([0.5], [ms(R[n])[0]], yerr=[ms(R[n])[1]], fmt=mk, color=col, ms=5, capsize=2, label=lab)
            ax[1].errorbar([0.5], [ms(R[n], "R@5", "retrieval")[0] * 100], yerr=[ms(R[n], "R@5", "retrieval")[1] * 100],
                           fmt=mk, color=col, ms=5, capsize=2, label=lab)
    ax[0].set_xlabel("contrastive weight $\\lambda$ (0 = classification only)"); ax[0].set_ylabel("macro AUROC")
    ax[0].set_title("(a) Diagnosis", loc="left")
    ax[1].axhline(100 * 5 / 120, color=INK2, lw=1, ls=":")
    ax[1].text(1.0, 100 * 5 / 120 + 0.3, "chance ($\\approx$4%)", ha="right", fontsize=6.5, color=INK2)
    ax[1].set_xlabel("contrastive weight $\\lambda$"); ax[1].set_ylabel("image$\\to$note R@5 (%)")
    ax[1].set_title("(b) Cross-modal retrieval (test folds)", loc="left"); ax[1].set_ylim(bottom=0)
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2, rect=(0, 0.1, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.savefig(f"{FIG}/contrastive.pdf"); plt.close(fig)


def fig_perclass(R):
    names = [n for n in ["img_only", "txt_tab_only", "early_fusion", "late_fusion", "irene_k2", "clip_l0.5"] if n in R]
    labels = {"img_only": "Image only", "txt_tab_only": "Note+structured", "early_fusion": "Early fusion",
              "late_fusion": "Late fusion", "irene_k2": "IRENE (k=2)", "clip_l0.5": "IRENE + CLIP"}
    M = np.array([[np.mean([s["pooled"]["per_class_auroc"][l] for s in R[n]["seeds"].values()]) for l in LABELS] for n in names])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]), vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(7), ["COVID-19", "Viral (oth.)", "Bacterial", "Fungal", "Pneum. (oth.)", "TB", "No finding"],
                  fontsize=6.5, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), [labels[n] for n in names], fontsize=7.5)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > 0.8 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6.5); cb.set_label("AUROC", fontsize=7)
    fig.savefig(f"{FIG}/per_class.pdf"); plt.close(fig)


def fig_roc(R, y):
    names = [n for n in ["img_only", "txt_tab_only", "late_fusion", "irene_k2"] if n in R]
    labs = {"img_only": "Image only", "txt_tab_only": "Note + structured", "late_fusion": "Late fusion", "irene_k2": "IRENE (k=2)"}
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for a, c in zip(ax, [0, 2, 3]):
        for j, n in enumerate(names):
            p = mean_oof(n)
            fpr, tpr, _ = roc_curve(y[:, c], p[:, c])
            a.plot(fpr, tpr, color=C[j], lw=1.6, label=f"{labs[n]} ({roc_auc_score(y[:, c], p[:, c]):.2f})")
        a.plot([0, 1], [0, 1], color=GRID, lw=1)
        a.set_title(f"{LABELS[c]} (n={int(y[:, c].sum())})", loc="left"); a.set_xlabel("false positive rate")
        a.legend(fontsize=6, loc="lower right"); a.set_aspect("equal")
    ax[0].set_ylabel("true positive rate")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f"{FIG}/roc.pdf"); plt.close(fig)


def fig_robust(R):
    pairs = [("txt_tab_only", "txt_tab_only_rawtext", "Note + structured"), ("irene_k2", "irene_k2_rawtext", "IRENE (k=2)")]
    pairs = [p for p in pairs if p[0] in R and p[1] in R]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.5]})
    if pairs:
        x = np.arange(len(pairs)); w = 0.36
        for j, (suffix, lab) in enumerate([(0, "diagnosis terms masked (default)"), (1, "raw notes (terms visible)")]):
            m = [ms(R[p[suffix]])[0] for p in pairs]; s = [ms(R[p[suffix]])[1] for p in pairs]
            ax[0].bar(x + (j - 0.5) * w, m, w * 0.92, yerr=s, color=C[j], label=lab, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK2})
            for xx, mm in zip(x + (j - 0.5) * w, m):
                ax[0].text(xx, mm + 0.012, f"{mm:.3f}", ha="center", fontsize=7)
        ax[0].set_xticks(x, [p[2] for p in pairs]); ax[0].set_ylim(0.5, 1.02); ax[0].set_ylabel("macro AUROC")
        ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].set_title("(a) Label leakage through notes", loc="left")
        ax[0].grid(axis="x", visible=False)
    names = [n for n in ["img_only", "early_fusion", "late_fusion", "irene_k2", "irene_k2_moddrop"] if n in R]
    labs = {"img_only": "Image\nonly", "early_fusion": "Early", "late_fusion": "Late", "irene_k2": "IRENE", "irene_k2_moddrop": "IRENE +\nmod. dropout"}
    x = np.arange(len(names)) * 1.2; w = 0.46
    full = [ms(R[n])[0] for n in names]
    nt = [ms(R[n], sub="notext")[0] if n != "img_only" else ms(R[n])[0] for n in names]
    ax[1].bar(x - w / 2, full, w * 0.92, color=C[0], label="full input")
    ax[1].bar(x + w / 2, nt, w * 0.92, color=C[1], label="note removed at test")
    for xx, a_, b_ in zip(x, full, nt):
        ax[1].text(xx - w / 2, a_ + 0.008, f"{a_:.2f}", ha="center", fontsize=6)
        ax[1].text(xx + w / 2, b_ + 0.008, f"{b_:.2f}", ha="center", fontsize=6)
    ax[1].set_xticks(x, [labs[n] for n in names], fontsize=7); ax[1].set_ylim(0.5, 0.9)
    ax[1].set_ylabel("macro AUROC"); ax[1].legend(fontsize=6.5, loc="upper left", ncol=2)
    ax[1].set_title("(b) Missing-modality robustness", loc="left"); ax[1].grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2)
    fig.savefig(f"{FIG}/robustness.pdf"); plt.close(fig)


def fig_curves(R):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1))
    for j, n in enumerate([x for x in ["img_only", "early_fusion", "late_fusion", "irene_k2", "clip_l0.5"] if x in R]):
        H = [h for s in R[n]["seeds"].values() for h in s["hist"].values()]
        L = max(len(h) for h in H)
        for a, key in zip(ax, ["train_bce", "val_bce"]):
            M = np.full((len(H), L), np.nan)
            for i, h in enumerate(H):
                M[i, :len(h)] = [r[key] for r in h]
            a.plot(np.arange(1, L + 1), np.nanmean(M, 0), color=C[j], lw=1.6, label=n.replace("_", " "))
    ax[0].set_title("(a) Training BCE", loc="left"); ax[1].set_title("(b) Validation BCE (inner split)", loc="left")
    for a in ax:
        a.set_xlabel("epoch")
    ax[0].legend(fontsize=6.5)
    fig.tight_layout(w_pad=2)
    fig.savefig(f"{FIG}/curves.pdf"); plt.close(fig)


def fig_throughput():
    p = "results/throughput.json"
    if not os.path.exists(p):
        return None
    T = json.load(open(p))
    L = T["loader"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.6, 1]})
    confs = [(0, False, False), (1, False, False), (2, False, False), (2, True, True)]
    lab = ["0 workers", "1 worker", "2 workers", "2 workers\n+pin+persistent"]
    x = np.arange(len(confs)); w = 0.36
    for j, ds in enumerate(["decode", "mmap_cache"]):
        v = []
        for wk, pin, per in confs:
            r = [e for e in L if e["dataset"] == ds and e["workers"] == wk and e["pin_memory"] == pin and e["persistent"] == per]
            v.append(r[0]["samples_per_sec"] if r else np.nan)
        ax[0].bar(x + (j - 0.5) * w, v, w * 0.92, color=C[j], label={"decode": "decode + resize per sample", "mmap_cache": "pre-resized uint8 memmap"}[ds])
        for xx, vv in zip(x + (j - 0.5) * w, v):
            ax[0].text(xx, vv * 1.08, f"{vv:.0f}", ha="center", fontsize=6.5)
    ax[0].set_yscale("log"); ax[0].set_xticks(x, lab, fontsize=7); ax[0].set_ylabel("samples / s (log)")
    ax[0].legend(fontsize=6.5, loc="upper right", ncol=1); ax[0].set_title("(a) Input pipeline", loc="left"); ax[0].grid(axis="x", visible=False)
    ax[0].set_ylim(top=max(e["samples_per_sec"] for e in L) * 60)
    items = [("fp32", T.get("train_fp32_samples_per_sec")), (f"AMP ({T.get('amp_dtype', 'bf16')})", T.get("train_amp_samples_per_sec"))]
    for k in sorted(T):
        if k.startswith("ddp_world") and k.endswith("global_samples_per_sec"):
            items.append((f"DDP x{k.split('world')[1].split('_')[0]} (gloo)", T[k]))
    names_, vals = zip(*[i for i in items if i[1] is not None])
    ax[1].bar(range(len(vals)), vals, 0.6, color=[C[0], C[1], C[2], C[3]][:len(vals)])
    for i, v in enumerate(vals):
        ax[1].text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=7)
    ax[1].set_xticks(range(len(vals)), names_, fontsize=7); ax[1].set_ylabel("train samples / s")
    ax[1].set_title("(b) End-to-end training step", loc="left"); ax[1].grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2)
    fig.savefig(f"{FIG}/throughput.pdf"); plt.close(fig)
    return T


# ------------------------------------------------------------------ tables
def table_main(R, y, pat):
    rows = [("Image only", "img_only"), ("Structured only", "tab_only"), ("Note only", "txt_only"),
            ("Note + structured", "txt_tab_only"), ("Image + structured (k=2)", "img_tab"), (None, None),
            ("Early fusion (k=0)", "early_fusion"), ("Late fusion (k=6)", "late_fusion"),
            ("IRENE mid fusion (k=2)", "irene_k2"), ("IRENE + CLIP ($\\lambda$=0.5)", "clip_l0.5"),
            ("IRENE + SigLIP ($\\lambda$=0.5)", "siglip_l0.5")]
    avail = [r for r in rows if r[1] is None or r[1] in R]
    best = max(ms(R[n])[0] for _, n in avail if n)
    lines = []
    for lab, n in avail:
        if n is None:
            lines.append("\\midrule")
            continue
        d = R[n]
        ci = boot_ci(y, pat, mean_oof(n))
        a, s = ms(d)
        lines.append(f"{lab} & {fmt(a, s, abs(a - best) < 1e-9)} & [{ci[0]:.3f}, {ci[1]:.3f}] & "
                     f"{fmt(*ms(d, 'macro_auprc'))} & {fmt(*ms(d, 'macro_f1'))} & {fmt(*ms(d, 'accuracy'))} & "
                     f"{d['n_params'] / 1e6:.2f} \\\\")
    open(f"{TAB}/main.tex", "w").write("\n".join(lines) + "\n")


def table_significance(R, y, pat):
    ref = "irene_k2"
    comps = [("Image only", "img_only"), ("Note + structured", "txt_tab_only"), ("Early fusion", "early_fusion"),
             ("Late fusion", "late_fusion"), ("No cross-attn (k=2)", "attn_none"), ("IRENE + CLIP", "clip_l0.5")]
    out, lines = {}, []
    if ref not in R:
        return out
    pr = mean_oof(ref)
    for lab, n in comps:
        if n not in R:
            continue
        d, lo, hi, p = paired_bootstrap(y, pat, pr, mean_oof(n))
        out[n] = {"diff": d, "lo": lo, "hi": hi, "p": p}
        lines.append(f"vs.\\ {lab} & {d:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {p:.3f} \\\\")
    open(f"{TAB}/significance.tex", "w").write("\n".join(lines) + "\n")
    return out


def table_ablation(R):
    rows = []
    groups = [
        ("Fusion position", [("k=0 (early)", "early_fusion"), ("k=1", "irene_k1"), ("k=2 (IRENE)", "irene_k2"),
                             ("k=3", "irene_k3"), ("k=4", "irene_k4"), ("k=6, bidirectional", "irene_k6"),
                             ("k=6, no cross-attn (late)", "late_fusion")]),
        ("Attention structure (k=2)", [("self only (no exchange)", "attn_none"), ("image$\\to$text", "attn_i2t"),
                                       ("text$\\to$image", "attn_t2i"), ("co-attention (cross only)", "attn_bi_cross"),
                                       ("gated cross-attn", "attn_gated"), ("IRENE (self+cross)/2", "irene_k2")]),
        ("Alignment objective (k=2)", [("none", "irene_k2"), ("CLIP $\\lambda$=0.1", "clip_l0.1"), ("CLIP $\\lambda$=0.5", "clip_l0.5"),
                                       ("CLIP $\\lambda$=1.0", "clip_l1.0"), ("SigLIP $\\lambda$=0.1", "siglip_l0.1"),
                                       ("SigLIP $\\lambda$=0.5", "siglip_l0.5"), ("SigLIP $\\lambda$=1.0", "siglip_l1.0"),
                                       ("CLIP $\\lambda$=0.5, early (k=0)", "early_clip_l0.5"),
                                       ("CLIP $\\lambda$=0.5, late (k=6)", "late_clip_l0.5")]),
        ("Note input and training (k=2)", [("default (word tokens + doc token, masked)", "irene_k2"),
                                            ("without document token", "irene_k2_nodoc"),
                                            ("raw notes (diagnosis terms visible)", "irene_k2_rawtext"),
                                            ("modality dropout $p$=0.3", "irene_k2_moddrop")]),
    ]
    for g, items in groups:
        items = [i for i in items if i[1] in R]
        if not items:
            continue
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{g}}}}} \\\\")
        for lab, n in items:
            d = R[n]
            ret = "--"
            if d["config"]["contrastive"] != "none":
                ret = f"{ms(d, 'R@5', 'retrieval')[0] * 100:.1f}"
            nt = ms(d, sub="notext")
            rows.append(f"\\quad {lab} & {fmt(*ms(d))} & {fmt(*ms(d, 'macro_auprc'))} & {fmt(*nt)} & {ret} & {len(d['seeds'])} \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows = rows[:-1]
    open(f"{TAB}/ablation.tex", "w").write("\n".join(rows) + "\n")


def write_numbers(R, summary):
    """LaTeX macros so every number quoted in the paper text comes from the results files."""
    st = json.load(open("data/processed/stats.json"))
    samples = pickle.load(open("data/processed/covidcxr_mm.pkl", "rb"))
    L = ["% auto-generated by scripts/make_figures.py -- do not edit",
         "\\makeatletter",
         "\\newcommand{\\AUC}[1]{\\@nameuse{auc@#1}}", "\\newcommand{\\AUCsd}[1]{\\@nameuse{aucsd@#1}}",
         "\\newcommand{\\AUPRC}[1]{\\@nameuse{auprc@#1}}", "\\newcommand{\\NT}[1]{\\@nameuse{nt@#1}}",
         "\\newcommand{\\RFIVE}[1]{\\@nameuse{rfive@#1}}", "\\newcommand{\\RONE}[1]{\\@nameuse{rone@#1}}",
         "\\newcommand{\\RTEN}[1]{\\@nameuse{rten@#1}}", "\\newcommand{\\PC}[2]{\\@nameuse{pc@#1@#2}}",
         "\\newcommand{\\SIG}[2]{\\@nameuse{sig@#1@#2}}", "\\newcommand{\\MIX}[1]{\\@nameuse{mix@#1}}",
         "\\newcommand{\\CL}[1]{\\@nameuse{cl@#1}}"]
    for n, d in summary.items():
        if n.startswith("_"):
            continue
        L.append(f"\\@namedef{{auc@{n}}}{{{d['auroc'][0]:.3f}}}\\@namedef{{aucsd@{n}}}{{{d['auroc'][1]:.3f}}}"
                 f"\\@namedef{{auprc@{n}}}{{{d['auprc'][0]:.3f}}}")
        if d["notext"]:
            L.append(f"\\@namedef{{nt@{n}}}{{{d['notext'][0]:.3f}}}")
        if d["retrieval"]:
            L.append(f"\\@namedef{{rfive@{n}}}{{{100 * d['retrieval']['R@5'][0]:.1f}}}"
                     f"\\@namedef{{rone@{n}}}{{{100 * d['retrieval']['R@1'][0]:.1f}}}"
                     f"\\@namedef{{rten@{n}}}{{{100 * d['retrieval']['R@10'][0]:.1f}}}")
        for i, l in enumerate(LABELS):
            L.append(f"\\@namedef{{pc@{n}@{i}}}{{{d['per_class'][l]:.2f}}}")
        L.append(f"\\@namedef{{mix@{n}}}{{{d['mixed'][0]:.3f}}}")
    for n, v in (summary.get("_significance") or {}).items():
        L.append(f"\\@namedef{{sig@{n}@diff}}{{{v['diff']:+.3f}}}\\@namedef{{sig@{n}@lo}}{{{v['lo']:+.3f}}}"
                 f"\\@namedef{{sig@{n}@hi}}{{{v['hi']:+.3f}}}\\@namedef{{sig@{n}@p}}{{{v['p']:.3f}}}")
    for k, v in (summary.get("_classical") or {}).items():
        if isinstance(v, float):
            L.append(f"\\@namedef{{cl@{k}}}{{{v:.3f}}}")
    L.append("\\makeatother")
    texts = [s["text_masked"] for s in samples.values()]
    mx = max(d["config"]["max_txt"] for d in R.values()) if R else 48
    macros = {"RawRows": st["raw_rows"], "XrayRows": st["xray_rows"], "NImages": st["labelled_rows"],
              "NPatients": st["patients"], "NotesWith": st["with_notes"], "NotesMasked": st["notes_with_masked_term"],
              "PctMasked": f"{100 * st['notes_with_masked_term'] / st['with_notes']:.0f}",
              "NMaskedTerms": st["masked_terms_total"], "DateNotes": sum("[DATE]" in t for t in texts),
              "MaxTxt": mx, "NMixed": int(mixed_mask(samples).sum()), "NRuns": len(R), "NSeedsMain": 3,
              "NFoldRuns": sum(len(d["seeds"]) * 5 for d in R.values()),
              "ParamMin": f"{min(d['n_params'] for d in R.values()) / 1e6:.2f}" if R else "0",
              "ParamMax": f"{max(d['n_params'] for d in R.values()) / 1e6:.2f}" if R else "0",
              "CPUHours": f"{sum(d['wall_sec'] for d in R.values()) / 3600:.1f}"}
    T = summary.get("_throughput")
    if T:
        dec = max(e["samples_per_sec"] for e in T["loader"] if e["dataset"] == "decode")
        dec0 = [e["samples_per_sec"] for e in T["loader"] if e["dataset"] == "decode" and e["workers"] == 0][0]
        mm = max(e["samples_per_sec"] for e in T["loader"] if e["dataset"] == "mmap_cache")
        macros.update({"TpDecodeZero": f"{dec0:.0f}", "TpDecodeBest": f"{dec:.0f}", "TpMmapBest": f"{mm:.0f}",
                       "TpSpeedup": f"{mm / dec0:.0f}", "TpWorkerSpeedup": f"{dec / dec0:.1f}",
                       "TpFp": f"{T['train_fp32_samples_per_sec']:.1f}", "TpAmp": f"{T['train_amp_samples_per_sec']:.1f}",
                       "TpCacheSec": f"{T['cache_build_sec']:.0f}"})
        for k, v in T.items():
            if k.startswith("ddp_world") and k.endswith("global_samples_per_sec"):
                macros["TpDDP"] = f"{v:.1f}"
    for k, v in macros.items():
        L.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    open("paper/numbers.tex", "w").write("\n".join(L) + "\n")


def table_reference(R, summary):
    """Classical shortcut probes vs deep models, on all images and on the mixed-source subset."""
    if not os.path.exists("results/classical.json"):
        return
    c = json.load(open("results/classical.json"))
    rows = [("Source host only (one-hot)", "source_only"), ("View only", "view_only"),
            ("All structured fields", "structured_all"), ("TF-IDF note, [DX]-masked", "tfidf_masked"),
            ("TF-IDF note, raw", "tfidf_raw")]
    L = ["\\multicolumn{3}{l}{\\textit{Logistic regression probes}} \\\\"]
    for lab, k in rows:
        L.append(f"\\quad {lab} & {c[k]:.3f} & {c[k + '@mixed']:.3f} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{3}{l}{\\textit{IRENE-MM variants (mean over seeds)}} \\\\")
    for lab, n in [("Image only", "img_only"), ("Note + structured", "txt_tab_only"), ("Early fusion", "early_fusion"),
                   ("Late fusion", "late_fusion"), ("IRENE (k=2)", "irene_k2"), ("IRENE + CLIP", "clip_l0.5")]:
        if n in summary:
            L.append(f"\\quad {lab} & {summary[n]['auroc'][0]:.3f} & {summary[n]['mixed'][0]:.3f} \\\\")
    open(f"{TAB}/reference.tex", "w").write("\n".join(L) + "\n")


def main():
    R = load_runs()
    samples, keys, y, pat = store()
    mix = mixed_mask(samples)
    fig_dataset(samples, y)
    summary = {n: {"auroc": ms(d), "auprc": ms(d, "macro_auprc"), "f1": ms(d, "macro_f1"), "acc": ms(d, "accuracy"),
                   "notext": ms(d, sub="notext") if "notext" in next(iter(d["seeds"].values()))["pooled"] else None,
                   "retrieval": {k: ms(d, k, "retrieval") for k in ["R@1", "R@5", "R@10", "median_rank"]}
                   if d["config"]["contrastive"] != "none" else None,
                   "n_seeds": len(d["seeds"]), "n_params": d["n_params"], "wall_sec": d["wall_sec"],
                   "per_class": {l: float(np.mean([s["pooled"]["per_class_auroc"][l] for s in d["seeds"].values()])) for l in LABELS},
                   "epochs_mean": float(np.mean([len(h) for s in d["seeds"].values() for h in s["hist"].values()])),
                   "mixed": (float(np.mean(v := [macro_auc(y[mix], o[mix]) for o in seed_oofs(n)])), float(np.std(v)))}
               for n, d in R.items()}
    if R:
        fig_main(R); fig_fusion(R); fig_attn(R); fig_perclass(R); fig_roc(R, y); fig_robust(R); fig_curves(R)
        if any(k.startswith("clip") for k in R):
            fig_contrastive(R)
        table_main(R, y, pat); table_ablation(R)
        summary["_significance"] = table_significance(R, y, pat)
    table_reference(R, summary)
    if os.path.exists("results/classical.json"):
        summary["_classical"] = json.load(open("results/classical.json"))
    summary["_throughput"] = fig_throughput()
    write_numbers(R, summary)
    json.dump(summary, open("results/summary.json", "w"), indent=1)
    for n in sorted(R, key=lambda n: -summary[n]["auroc"][0]):
        s = summary[n]
        print(f"{n:24s} AUROC {s['auroc'][0]:.4f}±{s['auroc'][1]:.4f}  AUPRC {s['auprc'][0]:.3f}  "
              f"noText {s['notext'][0] if s['notext'] else float('nan'):.3f}  seeds {s['n_seeds']}  "
              f"R@5 {s['retrieval']['R@5'][0] if s['retrieval'] else float('nan'):.3f}")


if __name__ == "__main__":
    main()
