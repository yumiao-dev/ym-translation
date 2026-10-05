# -*- coding: utf-8 -*-
"""
ym-Translation 0.1B 中英翻译模型 —— 从零预训练

架构: Decoder-only, GQA(8Q/2KV) + QK-Norm + RoPE + SwiGLU + RMSNorm + tied embedding
       (与 yumiao05b 同源，已验证可跑)
面向: 端侧离线翻译（安卓），目标 > 机翻质量，4-8K 上下文

关键配置:
  VOCAB = 32000   (自训中英联合 BPE，embedding 仅 ~25M)
  CTX   = 4096    (端侧 4-8K 上下文)
  RoPE base = 500000 (长上下文友好，可外推到 8K)

数据: ym-Translation/data/tokens/part_*.bin  (uint32 流)
      样本格式: <|zh|> 中文 <|en|> English <|eos|>   (双向 + 句级/段落级混合)
"""
import os, sys, math, time, glob, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 超参 ----------------
HIDDEN = 512
LAYERS = 10
HEADS = 8
KVHEADS = 2
HEAD_DIM = HIDDEN // HEADS          # 64
FFN = 2048
VOCAB = 32000                       # 新词表
CTX = 4096                          # 4K（RoPE 可外推 8K）
DROPOUT = 0.0
ROPE_BASE = 500000.0

LR_PEAK = 1e-3                      # 小词表小模型，可以稍大
LR_MIN = 1e-4
WARMUP = 100
MICRO_BS = 8                        # CTX 4K + logits，micro bs 降到 8 防 OOM
GRAD_ACCUM = 16                     # global batch = 8*4096*16 = 524288 tok/step
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "2000000000"))
TOTAL_STEPS = TARGET_TOKENS // (MICRO_BS * CTX * GRAD_ACCUM)
WEIGHT_DECAY = 0.1
BETA = (0.9, 0.95)
GRAD_CLIP = 1.0
MAX_SECONDS = float(os.environ.get("TRAIN_MAX_SECONDS", "18000"))   # 5h 安全线

BASE_DIR = "/mnt/workspace/ym-Translation"
CKPT_DIR = os.path.join(BASE_DIR, "output/ckpt")
LOG_PATH = "/tmp/train_trans.log"
DATA_GLOB = os.path.join(BASE_DIR, "data/tokens/part_*.bin")

CKPT_EVERY = 300                    # 每 300 步存一次（约 8 分钟）
CKPT_KEEP = 3                        # 保留最近 N 个检查点（防写入时被杀损坏）
os.makedirs(CKPT_DIR, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------- 模型 ----------------
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
        self.qn = nn.RMSNorm(HEAD_DIM)
        self.kn = nn.RMSNorm(HEAD_DIM)

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
        self.n1 = nn.RMSNorm(HIDDEN)
        self.attn = Attention()
        self.n2 = nn.RMSNorm(HIDDEN)
        self.up = nn.Linear(HIDDEN, 2 * FFN, bias=False)
        self.down = nn.Linear(FFN, HIDDEN, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        g, u = self.up(self.n2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(g) * u)
        return x


class YMTranslate(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.nf = nn.RMSNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.head.weight = self.emb.weight          # tied
        self.apply(self._init)
        for name, p in self.named_parameters():
            if 'emb.weight' in name:
                p.DO_NOT_WD = True

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = rope_cache(T, HEAD_DIM, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.nf(x)
        loss = None
        logits = None
        if targets is not None:
            # 训练时：分块计算（chunked cross-entropy），避免 [B,T,VOCAB] 巨矩阵爆显存
            loss = chunked_ce(self.head, x, targets)
        else:
            logits = self.head(x)
        return logits, loss


def chunked_ce(head, x, targets, chunk=1024):
    """分块交叉熵：把 (B*T) 维切成若干块，逐块算 head + CE
    显存峰值从 B*T*VOCAB 降到 chunk*VOCAB
    """
    B, T, C = x.shape
    xf = x.reshape(B * T, C)
    tf = targets.reshape(B * T)
    total = 0.0
    n = 0
    for i in range(0, B * T, chunk):
        xb = xf[i:i + chunk]
        tb = tf[i:i + chunk]
        lg = head(xb)                                   # [chunk, VOCAB]
        total = total + F.cross_entropy(lg.float(), tb, reduction='sum')
        n += tb.numel()
    return total / max(n, 1)


# ---------------- 数据 ----------------
class BinStream:
    """多 part 流式读取 uint32 token，随机起始，支持边训边等数据"""
    def __init__(self, dev='cuda'):
        self.dev = dev
        self._min = MICRO_BS * CTX * 4 * 8
        self.parts = sorted(p for p in glob.glob(DATA_GLOB)
                            if os.path.getsize(p) > self._min)
        self.fhs = [open(p, 'rb', buffering=0) for p in self.parts]
        log(f"data parts: {[os.path.basename(p) for p in self.parts]}")
        if not self.parts:
            raise RuntimeError("NO DATA PARTS in " + DATA_GLOB)

    def rescan(self):
        cur = set(self.parts)
        new = [p for p in sorted(glob.glob(DATA_GLOB))
               if p not in cur and os.path.getsize(p) > self._min]
        if new:
            for fh in self.fhs:
                fh.close()
            self.parts = sorted(glob.glob(DATA_GLOB))
            self.parts = [p for p in self.parts if os.path.getsize(p) > self._min]
            self.fhs = [open(p, 'rb', buffering=0) for p in self.parts]
            log(f"data rescan: +{len(new)} part(s)")
            return True
        return False

    def read_batch(self, bs, seq):
        need = bs * seq + 1
        out = np.empty(need, dtype=np.uint32)
        got = 0
        guard = 0
        while got < need:
            guard += 1
            if guard > 1000:
                raise RuntimeError("read_batch stuck")
            i = np.random.randint(0, len(self.fhs))
            fh = self.fhs[i]
            size = max(0, os.path.getsize(self.parts[i]) - 64 * 1024 * 1024) // 4
            span = size - need
            if span <= 0:
                continue
            start = np.random.randint(0, span)
            fh.seek(start * 4)
            nread = min(need - got, size - start)
            if nread <= 0:
                continue
            chunk = np.frombuffer(fh.read(nread * 4), dtype=np.uint32)
            if len(chunk) == 0:
                continue
            out[got:got + len(chunk)] = chunk
            got += len(chunk)
        x = torch.from_numpy(out[:-1].reshape(bs, seq).astype(np.int64))
        y = torch.from_numpy(out[1:].reshape(bs, seq).astype(np.int64))
        # 目标设备：CUDA / ROCm 均用 'cuda' 接口；pin_memory 失败则退化为直接拷贝
        if self.dev == 'cuda':
            try:
                return (x.pin_memory().to('cuda', non_blocking=True),
                        y.pin_memory().to('cuda', non_blocking=True))
            except Exception:
                pass
        return x.to(self.dev), y.to(self.dev)


# ---------------- 训练 ----------------
def get_lr(step):
    if step < WARMUP:
        return LR_PEAK * step / WARMUP
    p = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + math.cos(min(p, 1.0) * math.pi))


def _clean_sd(model):
    """取 state_dict 并剥掉 torch.compile 引入的 '_orig_mod.' 前缀，
    保证跨平台/跨 compile 状态都能互相加载"""
    sd = model.state_dict()
    if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


def save_ckpt(model, opt, step, tokens_done):
    """原子写入 + 轮转备份
    1. 先写 step_XXXXX.pt.tmp，再 rename（原子）
    2. 同时更新 latest.pt
    3. 只保留最近 CKPT_KEEP 个 step_*.pt
    """
    payload = {"model": _clean_sd(model), "opt": opt.state_dict(),
               "step": step, "tokens": tokens_done}
    # 1) 步编号检查点
    sfile = os.path.join(CKPT_DIR, f"step_{step:06d}.pt")
    tmp = sfile + ".tmp"
    torch.save(payload, tmp)
    os.rename(tmp, sfile)
    # 2) latest.pt（续训读这个）
    ltmp = os.path.join(CKPT_DIR, "latest.pt.tmp")
    torch.save(payload, ltmp)
    os.rename(ltmp, os.path.join(CKPT_DIR, "latest.pt"))
    # 3) 轮转：只留最近 CKPT_KEEP 个 step_*.pt
    stems = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")))
    for old in stems[:-CKPT_KEEP]:
        try:
            os.remove(old)
        except Exception:
            pass
    log(f"ckpt saved @ step {step} ({tokens_done/1e9:.3f}B tok) "
        f"[keep {min(len(stems), CKPT_KEEP)}]")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    # ---- 设备自动检测（CUDA / ROCm 统一走 'cuda' 接口）----
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    _is_rocm = bool(getattr(torch.version, "hip", None))
    try:
        _gpu_name = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    except Exception:
        _gpu_name = '?'
    log("=" * 60)
    log(f"device={dev} | backend={'ROCm/HIP' if _is_rocm else 'CUDA'} | gpu={_gpu_name}")
    model = YMTranslate().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"YM-Translate 0.1B | params={nparam/1e6:.1f}M | "
        f"vocab={VOCAB} ctx={CTX} layers={LAYERS}")
    log(f"target {TARGET_TOKENS/1e9:.2f}B tokens | total_steps={TOTAL_STEPS} | "
        f"global_batch={MICRO_BS*CTX*GRAD_ACCUM} tok/step")
    log("=" * 60)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if getattr(p, 'DO_NOT_WD', False) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    # fused AdamW：ROCm / 老版本可能不支持，自动回退
    _params = [{"params": decay, "weight_decay": WEIGHT_DECAY},
               {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(_params, lr=LR_PEAK, betas=BETA, eps=1e-8, fused=True)
        log("optimizer: AdamW(fused=True)")
    except Exception as e:
        log(f"fused AdamW unavailable ({e}), fallback to plain AdamW")
        opt = torch.optim.AdamW(_params, lr=LR_PEAK, betas=BETA, eps=1e-8)

    step, tokens_done = 0, 0
    # 续训：优先 latest.pt，失败则回退到最近的 step_*.pt
    cands = [os.path.join(CKPT_DIR, "latest.pt")]
    cands += sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")), reverse=True)
    for ck in cands:
        if not os.path.exists(ck):
            continue
        try:
            d = torch.load(ck, map_location=dev, weights_only=False)
            sd = d["model"]
            # 兼容 torch.compile 存下的检查点（键名带 "_orig_mod." 前缀）
            # 以及反过来（当前是 compile 模型、检查点未 compile）的情况
            keys = list(sd.keys())
            if keys and all(k.startswith("_orig_mod.") for k in keys):
                sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
                log("stripped '_orig_mod.' prefix from checkpoint")
            model.load_state_dict(sd)
            opt.load_state_dict(d["opt"])
            step, tokens_done = d["step"], d["tokens"]
            log(f"resumed from {os.path.basename(ck)} @ step {step} "
                f"({tokens_done/1e9:.3f}B tok)")
            break
        except Exception as e:
            log(f"resume failed on {os.path.basename(ck)}: {e}")
            continue

    # torch.compile：可用环境变量 NO_COMPILE=1 关闭（ROCm 上偶发问题）
    if os.environ.get("NO_COMPILE", "0") == "1":
        log("torch.compile skipped (NO_COMPILE=1)")
    else:
        try:
            model = torch.compile(model)
            log("torch.compile OK")
        except Exception as e:
            log(f"compile skipped: {e}")

    data = BinStream(dev)
    model.train()
    t0 = time.time()
    log("=== training starts ===")

    last_log = t0
    while step < TOTAL_STEPS:
        if time.time() - t0 > MAX_SECONDS:
            log(f"!! MAX_SECONDS ({MAX_SECONDS}) reached, stopping")
            break
        lr = get_lr(step)
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = data.read_batch(MICRO_BS, CTX)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            acc_loss += loss.item() / GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        step += 1
        tokens_done += MICRO_BS * CTX * GRAD_ACCUM

        now = time.time()
        if now - last_log > 20:
            el = now - t0
            log(f"step {step}/{TOTAL_STEPS} | loss {acc_loss:.4f} | lr {lr:.2e} | "
                f"{tokens_done/1e9:.3f}B tok | {tokens_done/el/1000:.0f}K tok/s | "
                f"{(now-t0)/60:.1f}min")
            last_log = now
        if step % CKPT_EVERY == 0:
            save_ckpt(model, opt, step, tokens_done)
        if step % 200 == 0:
            data.rescan()

    save_ckpt(model, opt, step, tokens_done)
    log(f"=== TRAINING DONE: step {step}, {tokens_done/1e9:.3f}B tokens ===")


if __name__ == "__main__":
    main()
