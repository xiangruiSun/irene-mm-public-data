"""Configurable IRENE-style multimodal Transformer for fusion ablations.

The original IRENE (Zhou et al., Nat. Biomed. Eng. 2023) tokenises every modality, runs
L_f "bidirectional multimodal attention" blocks in which the image stream and the clinical
(text + structured) stream each attend to themselves *and* to the other stream, and then
concatenates both streams for L - L_f standard self-attention blocks (L_f = 2 in the paper).

This module generalises that design along the three axes studied in the paper:

  * fusion position   ``fusion_layer`` k in [0, L]:
        k = 0  -> early fusion (single stream from the first layer, a.k.a. "merged attention")
        0<k<L  -> mid fusion (k dual-stream blocks, then L-k joint blocks)  [IRENE: k = 2]
        k = L  -> late fusion (fully separate encoders; pooled features are concatenated)
  * attention structure inside dual-stream blocks ``cross_mode``:
        'bi_avg'   IRENE: out = (self-attn + cross-attn)/2 for both streams
        'bi_cross' ViLBERT/LXMERT co-attention: each stream uses cross-attn only
        'i2t'      image queries text (text stream is self-attn only)
        't2i'      text queries image (image stream is self-attn only)
        'gated'    Flamingo-style tanh-gated cross-attn residual added to self-attn (both streams)
        'none'     no cross-modal exchange (independent streams until fusion)
  * image-text alignment objective ``contrastive`` in {'none', 'clip', 'siglip'} applied
    to pooled image / note embeddings at the fusion point ("align before fuse", ALBEF/BLIP).

Also supports masking-aware attention (padding + missing modalities), learned missing-value
embeddings for structured tokens, and modality dropout.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMConfig:
    hidden: int = 128
    heads: int = 4
    layers: int = 6
    mlp: int = 256
    dropout: float = 0.1
    attn_dropout: float = 0.1
    fusion_layer: int = 2
    cross_mode: str = "bi_avg"
    contrastive: str = "none"
    proj_dim: int = 128
    img_in: int = 1024
    n_img_tok: int = 49
    txt_in: int = 303
    max_txt: int = 64
    n_lab: int = 11
    num_classes: int = 7
    use_img: bool = True
    use_txt: bool = True
    use_tab: bool = True
    modality_dropout: float = 0.0
    doc_dim: int = 0               # >0: prepend one document-level note token (e.g. TF-IDF->SVD)
    pixel_input: bool = False      # end-to-end ViT patch embedding from raw 224x224 images
    patch: int = 16


class MHA(nn.Module):
    """Multi-head attention with separate query / key-value sources and a key padding mask."""

    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.dk, self.p = h, d // h, p
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)

    def forward(self, xq, xkv, kv_mask=None):
        B, Nq, D = xq.shape
        Nk = xkv.shape[1]
        q = self.q(xq).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k, v = self.kv(xkv).view(B, Nk, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if kv_mask is not None:                      # True = valid key
            attn_mask = kv_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.p if self.training else 0.0)
        return o.transpose(1, 2).reshape(B, Nq, D)


class FFN(nn.Module):
    def __init__(self, d, m, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, m), nn.GELU(), nn.Dropout(p), nn.Linear(m, d), nn.Dropout(p))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Standard pre-LN Transformer block (joint / single-stream)."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.hidden, eps=1e-6), nn.LayerNorm(c.hidden, eps=1e-6)
        self.attn, self.out = MHA(c.hidden, c.heads, c.attn_dropout), nn.Linear(c.hidden, c.hidden)
        self.drop = nn.Dropout(c.dropout)
        self.ffn = FFN(c.hidden, c.mlp, c.dropout)

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.drop(self.out(self.attn(h, h, mask)))
        return x + self.ffn(self.n2(x))


class DualBlock(nn.Module):
    """IRENE bidirectional multimodal attention block, generalised over ``cross_mode``."""

    def __init__(self, c):
        super().__init__()
        self.mode = c.cross_mode
        d = c.hidden
        self.ni, self.nt = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.self_i, self.self_t = MHA(d, c.heads, c.attn_dropout), MHA(d, c.heads, c.attn_dropout)
        if self.mode != "none":
            self.x_it = MHA(d, c.heads, c.attn_dropout)  # image queries -> text keys/values
            self.x_ti = MHA(d, c.heads, c.attn_dropout)  # text queries -> image keys/values
        if self.mode == "gated":
            self.g_i = nn.Parameter(torch.zeros(1))
            self.g_t = nn.Parameter(torch.zeros(1))
        self.out_i, self.out_t = nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(c.dropout)
        self.fi, self.ft = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.ffn_i, self.ffn_t = FFN(d, c.mlp, c.dropout), FFN(d, c.mlp, c.dropout)

    def forward(self, xi, xt, mi, mt, cross=True):
        hi, ht = self.ni(xi), self.nt(xt)
        si, st = self.self_i(hi, hi, mi), self.self_t(ht, ht, mt)
        m = self.mode
        if m == "none" or not cross:
            ai, at = si, st
        else:
            ci = self.x_it(hi, ht, mt)
            ct = self.x_ti(ht, hi, mi)
            if m == "bi_avg":
                ai, at = (si + ci) / 2, (st + ct) / 2
            elif m == "bi_cross":
                ai, at = ci, ct
            elif m == "i2t":
                ai, at = (si + ci) / 2, st
            elif m == "t2i":
                ai, at = si, (st + ct) / 2
            elif m == "gated":
                ai, at = si + torch.tanh(self.g_i) * ci, st + torch.tanh(self.g_t) * ct
            else:
                raise ValueError(m)
        xi = xi + self.drop(self.out_i(ai))
        xt = xt + self.drop(self.out_t(at))
        xi = xi + self.ffn_i(self.fi(xi))
        xt = xt + self.ffn_t(self.ft(xt))
        return xi, xt


class Embeddings(nn.Module):
    """IRENE tokeniser: image patches | note tokens | sex | age | structured values."""

    def __init__(self, c):
        super().__init__()
        d = c.hidden
        self.c = c
        if c.pixel_input:
            self.patch = nn.Conv2d(1, d, c.patch, c.patch)
            n_img = (224 // c.patch) ** 2
        else:
            self.patch = nn.Sequential(nn.LayerNorm(c.img_in), nn.Linear(c.img_in, d))
            n_img = c.n_img_tok
        self.pe_img = nn.Parameter(torch.randn(1, n_img, d) * 0.02)
        self.txt = nn.Sequential(nn.Linear(c.txt_in, d))
        self.pe_txt = nn.Parameter(torch.randn(1, c.max_txt, d) * 0.02)
        if c.doc_dim > 0:
            self.doc = nn.Sequential(nn.LayerNorm(c.doc_dim), nn.Linear(c.doc_dim, d))
            self.pe_doc = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.sex, self.age = nn.Linear(1, d), nn.Linear(1, d)
        self.pe_sex, self.pe_age = nn.Parameter(torch.randn(1, 1, d) * .02), nn.Parameter(torch.randn(1, 1, d) * .02)
        self.miss_demo = nn.Parameter(torch.randn(2, d) * 0.02)
        self.lab = nn.Linear(1, d)                      # shared value projection, as in IRENE
        self.pe_lab = nn.Parameter(torch.randn(1, c.n_lab, d) * 0.02)
        self.miss_lab = nn.Parameter(torch.randn(1, c.n_lab, d) * 0.02)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, img, txt, txt_len, demo, lab, lab_mask, doc=None):
        c = self.c
        B = txt.shape[0]
        if c.pixel_input:
            xi = self.patch(img).flatten(2).transpose(1, 2)
        else:
            xi = self.patch(img)
        xi = self.drop(xi + self.pe_img)
        mi = torch.ones(B, xi.shape[1], dtype=torch.bool, device=xi.device)

        xt = self.txt(txt) + self.pe_txt[:, :txt.shape[1]]
        mt_txt = torch.arange(txt.shape[1], device=txt.device)[None] < txt_len[:, None]
        if c.doc_dim > 0:
            xd = self.doc(doc)[:, None] + self.pe_doc
            xt = torch.cat([xd, xt], 1)
            mt_txt = torch.cat([(txt_len > 0)[:, None], mt_txt], 1)
        age, sex, age_m, sex_m = demo[:, 0:1], demo[:, 1:2], demo[:, 2:3], demo[:, 3:4]
        xa = age_m * self.age(age) + (1 - age_m) * self.miss_demo[0]
        xs = sex_m * self.sex(sex) + (1 - sex_m) * self.miss_demo[1]
        xa, xs = xa[:, None] + self.pe_age, xs[:, None] + self.pe_sex
        lm = lab_mask[..., None]
        xl = lm * self.lab(lab[..., None]) + (1 - lm) * self.miss_lab + self.pe_lab
        tab = torch.cat([xs, xa, xl], 1)
        mt_tab = torch.ones(B, tab.shape[1], dtype=torch.bool, device=txt.device)
        if not c.use_tab:
            mt_tab = torch.zeros_like(mt_tab)
        if not c.use_txt:
            mt_txt = torch.zeros_like(mt_txt)
        xt = self.drop(torch.cat([xt, tab], 1))
        mt = torch.cat([mt_txt, mt_tab], 1)
        return xi, mi, xt, mt, mt_txt


def masked_mean(x, m):
    m = m.float()[..., None]
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class IRENEMM(nn.Module):
    def __init__(self, c: MMConfig):
        super().__init__()
        self.c = c
        assert 0 <= c.fusion_layer <= c.layers
        assert c.use_img or c.use_txt or c.use_tab
        self.emb = Embeddings(c)
        self.dual = nn.ModuleList([DualBlock(c) for _ in range(c.fusion_layer)])
        self.joint = nn.ModuleList([Block(c) for _ in range(c.layers - c.fusion_layer)])
        self.norm = nn.LayerNorm(c.hidden, eps=1e-6)
        self.late = c.fusion_layer == c.layers
        if self.late:
            self.norm_t = nn.LayerNorm(c.hidden, eps=1e-6)
        n_streams = 2 if self.late else 1
        self.head = nn.Linear(n_streams * c.hidden, c.num_classes)
        if c.contrastive != "none":
            self.proj_i = nn.Sequential(nn.Linear(c.hidden, c.hidden), nn.GELU(), nn.Linear(c.hidden, c.proj_dim))
            self.proj_t = nn.Sequential(nn.Linear(c.hidden, c.hidden), nn.GELU(), nn.Linear(c.hidden, c.proj_dim))
            if c.contrastive == "clip":
                self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            else:  # SigLIP init from Zhai et al. 2023
                self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
                self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, img, txt, txt_len, demo, lab, lab_mask, drop_text=None, doc=None):
        c = self.c
        xi, mi, xt, mt, mt_txt = self.emb(img, txt, txt_len, demo, lab, lab_mask, doc)
        if drop_text is not None:                     # modality dropout / test-time removal
            keep = ~drop_text[:, None]
            n_txt = mt_txt.shape[1]
            mt = torch.cat([mt[:, :n_txt] & keep, mt[:, n_txt:]], 1)
            mt_txt = mt_txt & keep
        if not c.use_img or not (c.use_txt or c.use_tab):
            # unimodal baselines: drop the absent stream entirely and run a single stream
            x, m = (xt, mt) if not c.use_img else (xi, mi)
            for blk in list(self.dual) + list(self.joint):
                x = blk(x, m) if isinstance(blk, Block) else x
            h = masked_mean(self.norm(x), m)
            return {"logits": self.head(h), "feat": h}
        xi0, xt0 = xi, xt
        for blk in self.dual:
            xi, xt = blk(xi, xt, mi, mt)
        has_txt = mt_txt.any(1)
        if c.contrastive != "none":
            # ---- alignment on *unimodal* representations ("align before fuse", ALBEF).
            # Cross-attention would let the note embedding see the image (trivial matching), so the
            # dual-stream blocks are re-run with cross-attention disabled (shared weights).
            ui, ut = xi0, xt0
            if c.cross_mode == "none":
                ui, ut = xi, xt
            else:
                for blk in self.dual:
                    ui, ut = blk(ui, ut, mi, mt, cross=False)
            zi = masked_mean(ui, mi)
            zt = masked_mean(ut[:, :mt_txt.shape[1]], mt_txt)
        if self.late:
            h = torch.cat([masked_mean(self.norm(xi), mi) * float(c.use_img),
                           masked_mean(self.norm_t(xt), mt)], -1)
        else:
            x = torch.cat([xi, xt], 1)
            m = torch.cat([mi, mt], 1)
            for blk in self.joint:
                x = blk(x, m)
            h = masked_mean(self.norm(x), m)
        logits = self.head(h)
        out = {"logits": logits, "feat": h}
        if c.contrastive != "none":
            out["emb_i"] = F.normalize(self.proj_i(zi), dim=-1)
            out["emb_t"] = F.normalize(self.proj_t(zt), dim=-1)
            out["has_txt"] = has_txt
        return out

    def contrastive_loss(self, out, text_ids):
        """CLIP (InfoNCE) or SigLIP loss over samples that have a note. Images of the same
        patient often share one note, so identical notes are treated as positives (multi-positive
        targets) instead of false negatives."""
        v = out["has_txt"]
        if v.sum() < 2:
            return out["logits"].new_zeros(())
        ei, et, ids = out["emb_i"][v].float(), out["emb_t"][v].float(), text_ids[v]
        pos = (ids[:, None] == ids[None, :]).float()
        sim = ei @ et.t()
        if self.c.contrastive == "clip":
            s = self.logit_scale.exp().clamp(max=100) * sim
            tgt = pos / pos.sum(1, keepdim=True)
            return 0.5 * (torch.sum(-tgt * F.log_softmax(s, 1), 1).mean()
                          + torch.sum(-tgt.t() * F.log_softmax(s.t(), 1), 1).mean())
        s = self.logit_scale.exp() * sim + self.logit_bias
        lab = 2 * pos - 1
        return -F.logsigmoid(lab * s).sum() / s.shape[0]
