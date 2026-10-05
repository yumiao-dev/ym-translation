# -*- coding: utf-8 -*-
"""
ym-Translation 续训（Stage 2）—— 补口语/语气词/习语能力

从 step_003814（2.0B tokens）续训，目标是修 C 类（口语应答语 0/12）和 E 类（话语标记 2/10）。

与 train_trans.py 的差异：
  1. TARGET_TOKENS 放大到 4.0B（不是 2.0B）—— 否则 cosine 调度认为已训完，LR 停在谷底
  2. LR_PEAK 降到 3e-4（续训用小 LR，避免破坏已学到的能力）
  3. WARMUP 重设为 60（续训也要爬坡）
  4. 数据源扩大：data/tokens/*.bin + data/tokens2/*.bin（新语料分词后放这里）
  5. 检查点写入 output/ckpt2/（不覆盖原来那批）
  6. 支持数据配比：新语料重复 N 遍（oversample）
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
HEAD_DIM = HIDDEN // HEADS
FFN = 2048
VOCAB = 32000
CTX = 4096
DROPOUT = 0.0
ROPE_BASE = 500000.0

# ---- 续训专用 ----
TOTAL_TOKENS_TARGET = int(os.environ.get("TARGET_TOKENS", "4000000000"))  # 4.0B（含已训 2.0B）
LR_PEAK = float(os.environ.get("LR_PEAK", "3e-4"))
LR_MIN = float(os.environ.get("LR_MIN", "3e-5"))
WARMUP = int(os.environ.get("WARMUP", "60"))
MICRO_BS = 8
GRAD_ACCUM = 16
TOTAL_STEPS = TOTAL_TOKENS_TARGET // (MICRO_BS * CTX * GRAD_ACCUM)
WEIGHT_DECAY = 0.1
BETA = (0.9, 0.95)
GRAD_CLIP = 1.0
MAX_SECONDS = float(os.environ.get("TRAIN_MAX_SECONDS", "20000"))

BASE_DIR = "/mnt/workspace/ym-Translation"
CKPT_DIR = os.path.join(BASE_DIR, "output/ckpt2")     # 新检查点目录
SRC_CKPT_DIR = os.path.join(BASE_DIR, "output/ckpt")  # 旧检查点（只读）
LOG_PATH = "/tmp/train_trans2.log"
DATA_GLOBS = [
    os.path.join(BASE_DIR, "data/tokens/part_*.bin"),   # 原有 2B
    # 新增口语/习语：tokens2/ 下实际是 ord_*.bin(44) + prag_*.bin(20)
    os.path.join(BASE_DIR, "data/tokens2/*.bin"),
]
# 新数据（语气词/习语/口语）oversample 权重：0=关掉, 2=重复2次
NEW_DATA_REPEAT = int(os.environ.get("NEW_DATA_REPEAT", "2"))
CKPT_EVERY = 250
CKPT_KEEP = 4
os.makedirs(CKPT_DIR, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------- 模型（与 train_trans.py 完全一致，保证权重可加载）----------------
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
        self.head.weight = self.emb.weight
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
            loss = chunked_ce(self.head, x, targets)
        else:
            logits = self.head(x)
        return logits, loss


def chunked_ce(head, x, targets, chunk=1024):
    B, T, C = x.shape
    xf = x.reshape(B * T, C)
    tf = targets.reshape(B * T)
    total = 0.0
    n = 0
    for i in range(0, B * T, chunk):
        xb = xf[i:i + chunk]
        tb = tf[i:i + chunk]
        lg = head(xb)
        total = total + F.cross_entropy(lg.float(), tb, reduction='sum')
        n += tb.numel()
    return total / max(n, 1)


# ---------------- 数据（支持多目录 + 配比）----------------
class BinStream:
    """多目录流式读取；weight 控制采样概率（用于新语料 oversample）"""
    def __init__(self, dev='cuda'):
        self.dev = dev
        self._min = MICRO_BS * CTX * 4 * 8
        self.groups = []      # [(paths, weight)]
        self._load()
        if not any(g[0] for g in self.groups):
            raise RuntimeError("NO DATA PARTS in " + str(DATA_GLOBS))

    def _load(self):
        self.fhs = []
        for gi, g in enumerate(DATA_GLOBS):
            paths = sorted(p for p in glob.glob(g) if os.path.getsize(p) > self._min)
            w = 1.0
            # 新增语料目录（tokens2）→ oversample
            if "tokens2" in g:
                w = float(NEW_DATA_REPEAT)
                if w > 1:
                    paths = paths * int(w)      # 同一批文件重复注册 → 采样概率×N
                    w = 1.0
            tot_gb = sum(os.path.getsize(p) for p in paths) / 1024**3
            log(f"group{gi} {os.path.basename(os.path.dirname(g))}: "
                f"{len(paths)} parts | {tot_gb:.2f}GB | repeat={NEW_DATA_REPEAT if 'tokens2' in g else 1}")
            self.groups.append((paths, w))
            for p in paths:
                self.fhs.append(open(p, 'rb', buffering=0))

    def rescan(self):
        before = len(self.fhs)
        for fh in self.fhs:
            fh.close()
        self.groups = []
        self._load()
        return len(self.fhs) != before

    def read_batch(self, bs, seq):
        need = bs * seq + 1
        out = np.empty(need, dtype=np.uint32)
        got = 0
        guard = 0
        while got < need:
            guard += 1
            if guard > 2000:
                raise RuntimeError("read_batch stuck")
            i = np.random.randint(0, len(self.fhs))
            fh = self.fhs[i]
            size = max(0, os.path.getsize(fh.name) - 64 * 1024 * 1024) // 4
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
        if self.dev == 'cuda':
            try:
                return (x.pin_memory().to('cuda', non_blocking=True),
                        y.pin_memory().to('cuda', non_blocking=True))
            except Exception:
                pass
        return x.to(self.dev), y.to(self.dev)


def get_lr(step, start_step):
    """LR 调度基于【相对步数】—— 续训时从起点重新爬坡，而不是接着旧余弦曲线

    否则：step=3814 时 p≈0.47，LR 已在余弦中间，会立刻衰减到谷底。
    """
    rel = step - start_step
    if rel <= 0:
        return LR_PEAK * 0.1          # 起点给一个小 LR
    if rel < WARMUP:
        return LR_PEAK * rel / WARMUP
    p = (rel - WARMUP) / max(1, (TOTAL_STEPS - start_step) - WARMUP)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + math.cos(min(p, 1.0) * math.pi))


def _clean_sd(model):
    sd = model.state_dict()
    if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


def save_ckpt(model, opt, step, tokens_done):
    payload = {"model": _clean_sd(model), "opt": opt.state_dict(),
               "step": step, "tokens": tokens_done}
    sfile = os.path.join(CKPT_DIR, f"step_{step:06d}.pt")
    tmp = sfile + ".tmp"
    torch.save(payload, tmp)
    os.rename(tmp, sfile)
    ltmp = os.path.join(CKPT_DIR, "latest.pt.tmp")
    torch.save(payload, ltmp)
    os.rename(ltmp, os.path.join(CKPT_DIR, "latest.pt"))
    stems = sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")))
    for old in stems[:-CKPT_KEEP]:
        try:
            os.remove(old)
        except Exception:
            pass
    log(f"ckpt saved @ step {step} ({tokens_done/1e9:.3f}B tok) [keep {min(len(stems), CKPT_KEEP)}]")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    _is_rocm = bool(getattr(torch.version, "hip", None))
    try:
        _gpu_name = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    except Exception:
        _gpu_name = '?'
    log("=" * 60)
    log(f"STAGE2 续训 | device={dev} | backend={'ROCm/HIP' if _is_rocm else 'CUDA'} | gpu={_gpu_name}")
    model = YMTranslate().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"params={nparam/1e6:.1f}M | lr_peak={LR_PEAK} | target={TOTAL_TOKENS_TARGET/1e9:.2f}B | steps={TOTAL_STEPS}")
    log("=" * 60)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if getattr(p, 'DO_NOT_WD', False) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    _params = [{"params": decay, "weight_decay": WEIGHT_DECAY},
               {"params": no_decay, "weight_decay": 0.0}]
    try:
        opt = torch.optim.AdamW(_params, lr=LR_PEAK, betas=BETA, eps=1e-8, fused=True)
        log("optimizer: AdamW(fused=True)")
    except Exception as e:
        log(f"fused AdamW unavailable ({e}), fallback")
        opt = torch.optim.AdamW(_params, lr=LR_PEAK, betas=BETA, eps=1e-8)

    step, tokens_done = 0, 0
    # 优先新目录的 latest.pt（第二次续训），否则从旧目录接
    cands = ([os.path.join(CKPT_DIR, "latest.pt")] +
             sorted(glob.glob(os.path.join(CKPT_DIR, "step_*.pt")), reverse=True) +
             [os.path.join(SRC_CKPT_DIR, "latest.pt")] +
             sorted(glob.glob(os.path.join(SRC_CKPT_DIR, "step_*.pt")), reverse=True))
    for ck in cands:
        if not os.path.exists(ck):
            continue
        try:
            d = torch.load(ck, map_location=dev, weights_only=False)
            sd = d["model"]
            if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
                sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
                log("stripped '_orig_mod.' prefix")
            model.load_state_dict(sd)
            try:
                opt.load_state_dict(d["opt"])
            except Exception as e:
                log(f"optimizer state not loaded ({e}) —— 只加载权重，优化器重新开始")
            step, tokens_done = d["step"], d["tokens"]
            log(f"RESUMED from {ck} @ step {step} ({tokens_done/1e9:.3f}B tok)")
            break
        except Exception as e:
            log(f"resume failed on {os.path.basename(ck)}: {e}")
            continue

    if os.environ.get("NO_COMPILE", "0") == "1":
        log("torch.compile skipped")
    else:
        try:
            model = torch.compile(model)
            log("torch.compile OK")
        except Exception as e:
            log(f"compile skipped: {e}")

    data = BinStream(dev)
    model.train()
    START_STEP = step            # 续训起点，LR 调度从这里重新算
    log(f"START_STEP={START_STEP} | 余下 {max(0, TOTAL_STEPS - START_STEP)} 步 "
        f"({max(0, TOTAL_STEPS - START_STEP) * MICRO_BS * CTX * GRAD_ACCUM / 1e9:.2f}B tok)")
    t0 = time.time()
    log("=== stage2 training starts ===")

    last_log = t0
    while step < TOTAL_STEPS:
        if time.time() - t0 > MAX_SECONDS:
            log(f"!! MAX_SECONDS reached, stopping")
            break
        lr = get_lr(step, START_STEP)
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
    log(f"=== STAGE2 DONE: step {step}, {tokens_done/1e9:.3f}B tokens ===")


if __name__ == "__main__":
    main()
