"""Build IRENE-format multimodal samples from the public COVID-19 chest X-ray dataset
(Cohen et al., 2020; https://github.com/ieee8023/covid-chestxray-dataset).

Each sample is an (image, unstructured text, structured record, multi-label target) tuple,
stored in the same key/value layout the original IRENE code expects:

    sample['pdesc']  cleaned clinical note tokens      (unstructured text)
    sample['bics']   [age, sex]                        (demographics)
    sample['bts']    structured clinical values + mask (labs / vitals / acquisition)
    sample['label']  multi-hot disease vector

Pipeline steps
  1. filter to chest radiographs with a resolved diagnosis
  2. hierarchical finding string -> 7-way multi-hot label
  3. text cleaning: unicode NFKC, URL/figure-reference stripping, date & location
     de-identification, whitespace normalisation, optional masking of label-revealing terms
  4. structured features: z-scored continuous values with explicit missingness masks
  5. patient-grouped, label-stratified K-fold split (no patient appears in two folds)

Usage:
    python data/prepare.py --raw data/raw/covid-chestxray-dataset --out data/processed
"""
import argparse
import json
import os
import pickle
import re
import unicodedata
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ["COVID-19", "Viral (other)", "Bacterial", "Fungal",
          "Pneumonia (other/unspec.)", "Tuberculosis", "No Finding"]

# Continuous structured fields. Outcome fields (survival, ICU, intubation) and
# RT-PCR status are deliberately excluded: they are post-hoc or directly reveal the label.
NUMERIC = ["age", "offset", "temperature", "pO2_saturation",
           "leukocyte_count", "neutrophil_count", "lymphocyte_count"]
VIEWS = ["PA", "AP", "AP Supine", "L"]

# Terms that name the diagnosis / pathogen or strongly date the case to the pandemic.
LEAK_TERMS = [
    r"covid[\s\-]?19", r"covid", r"sars[\s\-]?cov[\s\-]?2", r"sars[\s\-]?cov", r"sars", r"mers[\s\-]?cov",
    r"mers", r"2019[\s\-]?ncov", r"ncov", r"coronavirus(es)?", r"corona", r"wuhan",
    r"tuberculo\w*", r"\btb\b", r"mycobacteri\w*", r"pneumocystis", r"\bpcp\b", r"\bpjp\b", r"jirovecii",
    r"aspergill\w*", r"fungal", r"fungus", r"legionell\w*", r"klebsiell\w*", r"streptococc\w*",
    r"pneumococc\w*", r"mycoplasm\w*", r"nocardi\w*", r"chlamydo?phil\w*", r"e\.?\s?coli", r"escherichia",
    r"staphylococc\w*", r"\bmrsa\b", r"varicella", r"chickenpox", r"herpes", r"influenza", r"h1n1",
    r"\bflu\b", r"lipoid", r"bacteri\w*", r"viral", r"virus\w*", r"pneumoni\w*",
]
LEAK_RE = re.compile("|".join(LEAK_TERMS), flags=re.IGNORECASE)
MONTHS = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
DATE_RE = re.compile(rf"\b{MONTHS}\.?\s+\d{{1,2}}(st|nd|rd|th)?,?\s*(\d{{4}})?|\b\d{{1,2}}\s+{MONTHS}\.?\s*(\d{{4}})?|\b(19|20)\d{{2}}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
FIG_RE = re.compile(r"\(?\b(fig(ure)?s?\.?|panel)\s*\d+[a-z]?\)?", re.IGNORECASE)


def finding_to_label(finding: str):
    y = np.zeros(len(LABELS), dtype=np.float32)
    f = finding.lower()
    if "covid-19" in f:
        y[0] = 1
    elif "viral" in f:
        y[1] = 1
    elif "bacterial" in f:
        y[2] = 1
    elif "fungal" in f:
        y[3] = 1
    elif f.startswith("pneumonia"):
        y[4] = 1
    elif "tuberculosis" in f:
        y[5] = 1
    elif "no finding" in f:
        y[6] = 1
    else:
        return None
    return y


def clean_text(text, mask_leak=True):
    if not isinstance(text, str) or not text.strip():
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = URL_RE.sub(" ", t)
    t = FIG_RE.sub(" ", t)
    t = DATE_RE.sub(" [DATE] ", t)
    if mask_leak:
        t = LEAK_RE.sub(" [DX] ", t)
    t = re.sub(r"[^\S\n]+", " ", t)
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/covid-chestxray-dataset")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(os.path.join(args.raw, "metadata.csv"))
    stats = {"raw_rows": len(df)}
    df = df[df.modality == "X-ray"]
    stats["xray_rows"] = len(df)
    df = df[~df.finding.isin(["todo", "Unknown"])].copy()
    df = df[df.filename.apply(lambda f: os.path.exists(os.path.join(args.raw, "images", f)))]
    labels = df.finding.apply(finding_to_label)
    df = df[labels.notna()].copy()
    df["y"] = labels[labels.notna()]
    stats["labelled_rows"] = len(df)
    stats["patients"] = int(df.patientid.nunique())

    # --- structured features: z-score with train-agnostic robust stats (median/IQR over all rows is
    # label-free so it introduces no label leakage; the model never sees targets here)
    num = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    med, iqr = num.median(), (num.quantile(0.75) - num.quantile(0.25)).replace(0, 1.0)
    z = ((num - med) / iqr).clip(-5, 5)
    mask = num.notna().astype(np.float32)
    z = z.fillna(0.0).astype(np.float32)
    view = df.view.replace({"AP Erect": "AP"})
    view_oh = np.stack([(view == v).astype(np.float32).values for v in VIEWS], 1)
    sex = df.sex.map({"M": 1.0, "F": 0.0})
    sex_mask = sex.notna().astype(np.float32)

    samples, notes_raw = {}, {}
    for i, (_, r) in enumerate(df.iterrows()):
        key = r.filename  # full filename: some stems exist as both .jpg and .png
        note = r.clinical_notes if isinstance(r.clinical_notes, str) else ""
        bts = np.concatenate([z.iloc[i].values, view_oh[i]]).astype(np.float32)
        bts_mask = np.concatenate([mask.iloc[i].values, np.ones(len(VIEWS), np.float32)]).astype(np.float32)
        samples[key] = {
            "patient": str(r.patientid),
            "image": os.path.join("images", r.filename),
            "text_raw": clean_text(note, mask_leak=False),
            "text_masked": clean_text(note, mask_leak=True),
            "bics": np.array([z.iloc[i]["age"], sex.iloc[i] if sex_mask.iloc[i] else 0.0,
                              mask.iloc[i]["age"], sex_mask.iloc[i]], dtype=np.float32),
            "bts": bts, "bts_mask": bts_mask,
            "label": r.y, "finding": r.finding, "view": r.view,
            # publication / case-repository host: used only for confounding analysis, never as input
            "source": urlparse(str(r.url)).netloc.replace("www.", ""),
        }
        notes_raw[key] = note

    keys = list(samples)
    assert len(keys) == len(df), "duplicate sample keys"
    y_idx = np.array([samples[k]["label"].argmax() for k in keys])
    groups = np.array([samples[k]["patient"] for k in keys])
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold = np.zeros(len(keys), dtype=int)
    for f, (_, te) in enumerate(sgkf.split(keys, y_idx, groups)):
        fold[te] = f
    for k, f in zip(keys, fold):
        samples[k]["fold"] = int(f)
    # sanity: no patient leakage across folds
    pf = pd.DataFrame({"p": groups, "f": fold}).groupby("p").f.nunique()
    assert (pf == 1).all(), "patient appears in more than one fold"

    Y = np.stack([samples[k]["label"] for k in keys])
    stats["label_counts"] = {l: int(c) for l, c in zip(LABELS, Y.sum(0))}
    stats["fold_label_counts"] = {int(f): {l: int(c) for l, c in zip(LABELS, Y[fold == f].sum(0))} for f in range(args.folds)}
    stats["with_notes"] = int(sum(bool(samples[k]["text_raw"]) for k in keys))
    stats["note_words_median"] = float(np.median([len(samples[k]["text_raw"].split()) for k in keys if samples[k]["text_raw"]]))
    stats["masked_terms_total"] = int(sum(samples[k]["text_masked"].count("[DX]") for k in keys))
    stats["notes_with_masked_term"] = int(sum("[DX]" in samples[k]["text_masked"] for k in keys))
    stats["missing_rate"] = {c: float(1 - mask[c].mean()) for c in NUMERIC}
    stats["missing_rate"]["sex"] = float(1 - sex_mask.mean())
    stats["views"] = view.value_counts().to_dict()
    stats["numeric_fields"] = NUMERIC + [f"view={v}" for v in VIEWS]
    stats["labels"] = LABELS

    with open(os.path.join(args.out, "covidcxr_mm.pkl"), "wb") as fh:
        pickle.dump(samples, fh)
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "fold_label_counts"}, indent=2))


if __name__ == "__main__":
    main()
