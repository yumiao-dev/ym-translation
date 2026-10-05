# -*- coding: utf-8 -*-
"""ym-Translation 评估 v4：英译中（en2zh）方向

复用 eval_v3.py 的 47 句语料，但方向反过来：
  输入 = eval_v3 里的「期望英文」
  参考 = eval_v3 里的「中文源句」

目的：验证模型的双向能力，尤其是英文口语/语气词能否还原成地道中文。
"""
import os, sys, time, glob
import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/mnt/workspace/ym-Translation/scripts")
import rmsnorm_shim  # noqa: F401
from train_trans import YMTranslate, CTX
from eval_v3 import CASES

BASE = "/mnt/workspace/ym-Translation"
_c = glob.glob(os.path.join(BASE, "**/tokenizer.json"), recursive=True)
TOK_PATH = _c[0] if _c else os.path.join(BASE, "tokenizer/tokenizer.json")
L_ZH, L_EN = "<|zh|>", "<|en|>"
STOP_MARKS = ["<|eos|>", "<eos>", "<|endoftext|>"]


def load(ckpt):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = YMTranslate().to(dev)
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = d["model"]
    if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    print(f"loaded {ckpt} | step={d.get('step')} | {d.get('tokens',0)/1e9:.3f}B tok", flush=True)
    return m, dev


def gen(model, tok, dev, text, max_new=160):
    """en2zh：prompt = <|en|> text <|zh|>"""
    prompt = f"{L_EN} {text} {L_ZH}"
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new):
            xin = x[:, -CTX:]
            logits, _ = model(xin)
            nxt = logits[0, -1].argmax().item()
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=dev)], dim=1)
            if nxt in STOP_IDS:
                break
    dt = time.time() - t0
    seg = tok.decode(out)
    # 截断到第一个停止标记
    hit = [seg.find(mk) for mk in STOP_MARKS]
    hit = [h for h in hit if h >= 0]
    if hit:
        seg = seg[:min(hit)]
    # 去掉可能混入的标签
    for mk in (L_ZH, L_EN):
        pos = seg.find(mk)
        if pos >= 0:
            seg = seg[:pos]
    return seg.strip(), dt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "output/ckpt2/latest.pt"))
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=len(CASES))
    a = ap.parse_args()

    tok = Tokenizer.from_file(TOK_PATH)
    STOP_IDS = set()
    for mk in STOP_MARKS:
        e = tok.encode(mk)
        if len(e.ids) == 1:
            STOP_IDS.add(e.ids[0])
    if not STOP_IDS:
        e = tok.encode("<|eos|>")
        if e.ids:
            STOP_IDS.add(e.ids[-1])
    print(f"stop ids: {STOP_IDS}", flush=True)

    model, dev = load(a.ckpt)
    cases = CASES[a.lo:a.hi]
    print(f"=== EN->ZH EVAL: {len(cases)} sentences ===", flush=True)
    total_t = 0.0
    for i, (cat, zh, en, note) in enumerate(cases, start=a.lo + 1):
        # 英文侧可能含 "/"，取第一个作为输入
        src = en.split(" / ")[0].strip()
        out, dt = gen(model, tok, dev, src)
        total_t += dt
        print(f"[{i:02d}] {cat} | {src}  ==>  {out}   ({dt:.2f}s)", flush=True)
    print(f"=== DONE | {len(cases)} sent | {total_t:.1f}s | avg {total_t/max(len(cases),1):.2f}s/sent ===", flush=True)
