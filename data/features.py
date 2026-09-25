"""Cache modality-specific input tokens for fast multimodal training.

Image  : chest X-ray -> frozen DenseNet-121 pretrained on 8 public CXR corpora
         (torchxrayvision 'densenet121-res224-all', Cohen et al. 2022) -> 7x7x1024 grid = 49 patch
         tokens. This is the "hybrid" (CNN-stem) variant of the ViT tokenizer that IRENE already
         supports via config.patches.grid. K augmented views are cached per image for training.
Text   : cleaned clinical note -> spaCy tokenizer -> 300-d static word vectors (en_core_web_md)
         + 3 flag channels ([DX] mask, [DATE] mask, OOV) -> up to L tokens. IRENE used per-token
         BERT embeddings of the chief complaint in exactly this "token feature" layout.

Outputs (npz, float16): img_feats [N, V, 49, 1024], txt_raw / txt_masked [N, L, 303] + lengths.
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def load_xray(path):
    img = Image.open(path).convert("L")
    return img


def build_transforms():
    base = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    aug = T.Compose([
        T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
    ])
    return base, aug


def to_xrv_range(x):
    # torchxrayvision expects single-channel images scaled to [-1024, 1024]
    return (x * 2.0 - 1.0) * 1024.0


@torch.no_grad()
def image_features(samples, raw_dir, n_aug, batch=16, seed=0):
    import torchxrayvision as xrv
    torch.manual_seed(seed)
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    base, aug = build_transforms()
    keys = list(samples)
    V = 1 + n_aug
    out = np.zeros((len(keys), V, 49, 1024), dtype=np.float16)
    t0 = time.time()
    for v in range(V):
        tf = base if v == 0 else aug
        for s in range(0, len(keys), batch):
            ks = keys[s:s + batch]
            x = torch.stack([tf(load_xray(os.path.join(raw_dir, samples[k]["image"]))) for k in ks])
            f = model.features(to_xrv_range(x))                      # [B,1024,7,7]
            f = torch.relu(f).flatten(2).transpose(1, 2)            # [B,49,1024]
            out[s:s + len(ks), v] = f.numpy().astype(np.float16)
        print(f"view {v + 1}/{V} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def text_features(samples, field, max_len):
    import spacy
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
    keys = list(samples)
    D = nlp.vocab.vectors.shape[1]
    out = np.zeros((len(keys), max_len, D + 3), dtype=np.float16)
    lens = np.zeros(len(keys), dtype=np.int32)
    oov = 0
    ntok = 0
    for i, k in enumerate(keys):
        text = samples[k][field].replace("[DX]", " DXTOKEN ").replace("[DATE]", " DATETOKEN ")
        toks = [t for t in nlp.tokenizer(text) if not t.is_space][:max_len]
        lens[i] = len(toks)
        for j, t in enumerate(toks):
            if t.text == "DXTOKEN":
                out[i, j, D] = 1
            elif t.text == "DATETOKEN":
                out[i, j, D + 1] = 1
            else:
                lex = nlp.vocab[t.lower_]
                if lex.has_vector:
                    out[i, j, :D] = lex.vector / (np.linalg.norm(lex.vector) + 1e-6)
                else:
                    out[i, j, D + 2] = 1
                    oov += 1
            ntok += 1
    print(f"{field}: {ntok} tokens, OOV rate {oov / max(ntok, 1):.3f}, truncated {(lens == max_len).mean():.2%}")
    return out, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--raw", default="data/raw/covid-chestxray-dataset")
    ap.add_argument("--n_aug", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    with open(os.path.join(args.processed, "covidcxr_mm.pkl"), "rb") as fh:
        samples = pickle.load(fh)
    txt_raw, len_raw = text_features(samples, "text_raw", args.max_len)
    txt_masked, len_masked = text_features(samples, "text_masked", args.max_len)
    img = image_features(samples, args.raw, args.n_aug)
    np.savez(os.path.join(args.processed, "features.npz"), keys=np.array(list(samples)), img=img,
             txt_raw=txt_raw, len_raw=len_raw, txt_masked=txt_masked, len_masked=len_masked)
    print("saved", img.shape, txt_raw.shape)


if __name__ == "__main__":
    main()
