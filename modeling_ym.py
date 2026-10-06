# -*- coding: utf-8 -*-
"""ym-Translation 模型定义（独立版，仅依赖 torch）

从训练脚本中抽离，方便他人直接加载权重做推理。

用法：
    import torch
    from safetensors.torch import load_file
    from modeling_ym import YMTranslate, build_prompt

    model = YMTranslate()
    model.load_state_dict(load_file("model.safetensors"), strict=False)
    model.eval()
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 架构常量（与 config.json 一致）----------------
HIDDEN = 512
LAYERS = 10
HEADS = 8
KVHEADS = 2
HEAD_DIM = HIDDEN // HEADS
FFN = 2048
VOCAB = 32000
CTX = 4096
ROPE_BASE = 500000.0
RMS_EPS = 1e-6

# ---------------- 特殊 token ----------------
L_ZH = "<|zh|>"
L_EN = "<|en|>"
EOS = "<|eos|>"


class RMSNorm(nn.Module):
    """无 bias、无均值的 RMSNorm（带 eps，与训练一致）"""

    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def rope_cache(seq, dim, device, base=ROPE_BASE):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).bfloat16(), torch.sin(freqs).bfloat16()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(HIDDEN, KVHEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(HIDDEN, KVHEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.qn = RMSNorm(HEAD_DIM)
        self.kn = RMSNorm(HEAD_DIM)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, T, KVHEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, T, KVHEADS, HEAD_DIM).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if KVHEADS != HEADS:
            rep = HEADS // KVHEADS
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(HIDDEN)
        self.attn = Attention()
        self.n2 = RMSNorm(HIDDEN)
        self.up = nn.Linear(HIDDEN, 2 * FFN, bias=False)
        self.down = nn.Linear(FFN, HIDDEN, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        g, u = self.up(self.n2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(g) * u)
        return x


class YMTranslate(nn.Module):
    """ym-Translation 主干（与训练时完全一致）"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.nf = RMSNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.head.weight = self.emb.weight          # tied
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        cos, sin = rope_cache(T, HEAD_DIM, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.nf(x)
        return self.head(x)


def build_prompt(text, direction="zh2en"):
    """构造输入提示词

    direction: 'zh2en' 中译英 / 'en2zh' 英译中
    """
    if direction == "zh2en":
        return f"{L_ZH} {text} {L_EN}"
    return f"{L_EN} {text} {L_ZH}"


@torch.no_grad()
def generate(model, tok, text, direction="zh2en", max_new=256, device="cpu",
             repetition_penalty=1.0, no_repeat_ngram_size=0):
    """贪心解码（带可选 repetition penalty，缓解循环失控）

    注意：本项目原始评估用的是纯贪心（repetition_penalty=1.0）。
    加 penalty 后的效果尚未系统评估，属于路线图 P0 项。
    """
    model.eval()
    ids = tok(build_prompt(text, direction), add_special_tokens=False)["input_ids"]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = None
    for t in range(max_new):
        logits = model(idx)[:, -1, :].float()
        if repetition_penalty != 1.0:
            for token in set(idx[0].tolist()):
                if logits[0, token] > 0:
                    logits[0, token] /= repetition_penalty
                else:
                    logits[0, token] *= repetition_penalty
        if no_repeat_ngram_size > 0 and idx.shape[1] >= no_repeat_ngram_size:
            seq = idx[0].tolist()
            n = no_repeat_ngram_size
            prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
            banned = set()
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n - 1]) == prefix:
                    banned.add(seq[i + n - 1])
            for b in banned:
                logits[0, b] = -float("inf")
        nxt = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        tid = nxt.item()
        if tid == eos_id:
            break
        if tok.decode([tid]).strip() == EOS:
            break
    out_ids = idx[0, len(ids):].tolist()
    return tok.decode(out_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    m = YMTranslate()
    n = sum(p.numel() for p in m.parameters())
    print(f"params = {n/1e6:.1f}M")
    x = torch.randint(0, VOCAB, (1, 16))
    print("forward ok:", tuple(m(x).shape))
