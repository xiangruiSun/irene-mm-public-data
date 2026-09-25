"""Run the full experiment grid of the paper (5-fold patient-grouped CV x 3 seeds per config).

    python scripts/run_experiments.py --parallel 2          # all groups
    python scripts/run_experiments.py --only main fusion     # subset

Each config writes results/runs/<name>.json; finished configs are skipped (resumable).
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BASE = ["--eval_missing_text", "--doc_dim", "128", "--pos_weight", "sqrt"]
SEEDS = {"main": "0 1 2", "fusion": "0 1", "attn": "0 1", "contrastive": "0 1", "robust": "0 1"}

EXPERIMENTS = {
    # --- modality ablation + fusion paradigms
    "main": {
        "img_only": "--use_txt false --use_tab false",
        "txt_tab_only": "--use_img false",
        "txt_only": "--use_img false --use_tab false",
        "tab_only": "--use_img false --use_txt false",
        "img_tab": "--use_txt false --fusion_layer 2",
        "early_fusion": "--fusion_layer 0",
        "irene_k2": "--fusion_layer 2 --cross_mode bi_avg",
        "late_fusion": "--fusion_layer 6 --cross_mode none",
    },
    # --- where to fuse (IRENE bidirectional attention, k dual-stream blocks)
    "fusion": {f"irene_k{k}": f"--fusion_layer {k} --cross_mode bi_avg" for k in [1, 3, 4, 6]},
    # --- attention structure in the dual-stream blocks (k = 2)
    "attn": {
        "attn_bi_cross": "--fusion_layer 2 --cross_mode bi_cross",
        "attn_i2t": "--fusion_layer 2 --cross_mode i2t",
        "attn_t2i": "--fusion_layer 2 --cross_mode t2i",
        "attn_gated": "--fusion_layer 2 --cross_mode gated",
        "attn_none": "--fusion_layer 2 --cross_mode none",
    },
    # --- image-text contrastive alignment in a shared embedding space
    "contrastive": {
        **{f"clip_l{l}": f"--fusion_layer 2 --contrastive clip --lam {l}" for l in ["0.1", "0.5", "1.0"]},
        **{f"siglip_l{l}": f"--fusion_layer 2 --contrastive siglip --lam {l}" for l in ["0.1", "0.5", "1.0"]},
        "late_clip_l0.5": "--fusion_layer 6 --cross_mode none --contrastive clip --lam 0.5",
        "early_clip_l0.5": "--fusion_layer 0 --contrastive clip --lam 0.5",
    },
    # --- shortcut / leakage analysis and missing-modality robustness
    "robust": {
        "irene_k2_rawtext": "--fusion_layer 2 --text_field raw",
        "txt_tab_only_rawtext": "--use_img false --text_field raw",
        "irene_k2_moddrop": "--fusion_layer 2 --modality_dropout 0.3",
        "irene_k2_nodoc": "--fusion_layer 2 --doc_dim 0",
    },
}


def run(name, extra, out, seeds):
    if os.path.exists(os.path.join(out, f"{name}.json")):
        return name, "skip"
    if subprocess.call(["pgrep", "-f", "--", f"--name {name} "], stdout=subprocess.DEVNULL) == 0:
        return name, "already running"
    cmd = [sys.executable, "train.py", "--name", name, "--out", out, *BASE, "--seeds", *seeds.split(), *extra.split()]
    with open(os.path.join(out, f"{name}.log"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    return name, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results/runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(n, e, SEEDS[g]) for g, d in EXPERIMENTS.items() if not args.only or g in args.only for n, e in d.items()]
    with ThreadPoolExecutor(args.parallel) as ex:
        for name, rc in ex.map(lambda j: run(j[0], j[1], args.out, j[2]), jobs):
            print(name, rc, flush=True)


if __name__ == "__main__":
    main()
