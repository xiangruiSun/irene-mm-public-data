"""Classical (non-deep) baselines on the same patient-grouped folds, used to probe shortcuts:
logistic regression on (a) acquisition view only, (b) all structured fields, (c) TF-IDF of the
masked notes, (d) TF-IDF of the raw notes (diagnosis terms visible).
Writes results/classical.json.
"""
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier

S = pickle.load(open("data/processed/covidcxr_mm.pkl", "rb"))
K = list(S)
Y = np.stack([S[k]["label"] for k in K])
F = np.array([S[k]["fold"] for k in K])


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1])]))


def cv(make_x, C=1.0):
    oof = np.zeros_like(Y)
    for f in range(5):
        tr, te = F != f, F == f
        Xtr, Xte = make_x(tr, te)
        clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
        clf.fit(Xtr, Y[tr])
        oof[te] = clf.predict_proba(Xte)
    return macro(Y, oof), oof


def tab(idx):
    return np.stack([np.concatenate([S[k]["bts"], S[k]["bts_mask"], S[k]["bics"]]) for k in np.array(K)[idx]])


def view(idx):
    return np.stack([S[k]["bts"][7:11] for k in np.array(K)[idx]])


def tfidf(field):
    def mk(tr, te):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        docs = np.array([S[k][field] for k in K])
        return v.fit_transform(docs[tr]), v.transform(docs[te])
    return mk


def source(idx_tr, idx_te):
    src = np.array([S[k]["source"] for k in K])
    cats = sorted(set(src[idx_tr]))
    oh = lambda idx: np.stack([[float(s == c) for c in cats] for s in src[idx]])
    return oh(idx_tr), oh(idx_te)


res, oofs = {}, {}
src = np.array([S[k]["source"] for k in K])
mixed = np.isin(src, ["radiopaedia.org", "eurorad.org"])
for name, fn, C in [("source_only", source, 1.0), ("view_only", lambda tr, te: (view(tr), view(te)), 1.0),
                    ("structured_all", lambda tr, te: (tab(tr), tab(te)), 1.0),
                    ("tfidf_masked", tfidf("text_masked"), 4.0), ("tfidf_raw", tfidf("text_raw"), 4.0)]:
    res[name], oofs[name] = cv(fn, C)
    res[name + "@mixed"] = macro(Y[mixed], oofs[name][mixed])
res["mixed_source_hosts"] = ["radiopaedia.org", "eurorad.org"]
res["n_mixed_source"] = int(mixed.sum())
res["mixed_label_counts"] = Y[mixed].sum(0).astype(int).tolist()
np.savez("results/classical_oof.npz", **oofs)
json.dump(res, open("results/classical.json", "w"), indent=1)
print(res)
