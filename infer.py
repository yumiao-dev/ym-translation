# -*- coding: utf-8 -*-
"""ym-Translation 独立推理脚本（CPU / GPU 均可）

用法:
    python3 infer.py "今天天气怎么样？"
    python3 infer.py --interactive
"""
import os, sys, argparse
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modeling_ym import YMTranslate, build_prompt

CTX = 4096


def find_weights(model_dir):
    """兼容两种布局：<dir>/model/model.safetensors（发布包）与 <dir>/model.safetensors（HF 仓）"""
    for rel in (("model", "model.safetensors"), ("model.safetensors",)):
        p = os.path.join(model_dir, *rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"在 {model_dir} 下找不到 model.safetensors。"
        f"请用 --model 指向权重目录，例如：\n"
        f'  huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights\n'
        f'  python3 infer.py --model ym-weights "今天天气怎么样？"'
    )


def load(model_dir=None, device=None):
    model_dir = model_dir or HERE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    m = YMTranslate()
    sd = load_file(find_weights(model_dir))
    m.load_state_dict(sd, strict=False)
    m = m.to(device).eval()
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer", "tokenizer.json"))
    return m, tok, device


def clean(h):
    """去掉残余特殊 token，并截断自复读"""
    for mk in ("<|eos|>", "<eos>", "<|zh|>", "<|en|>", "<|endoftext|>"):
        h = h.replace(mk, " ")
    h = " ".join(h.split()).strip()
    n = len(h)
    if n < 40:
        return h
    for i in range(20, n - 20):
        w = min(60, n - i)
        if w < 20:
            break
        if h[i:i + w] == h[:w]:
            return h[:i].strip()
    return h


@torch.no_grad()
def translate(text, m, tok, device, max_new=192, direction="zh2en"):
    eos = tok.token_to_id("<eos>")
    ids = tok.encode(build_prompt(text, direction)).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = list(ids)
    for _ in range(max_new):
        lg = m(x[:, -CTX:])
        if isinstance(lg, (tuple, list)):
            lg = lg[0]
        nxt = int(lg[0, -1, :].float().argmax())
        if nxt == eos:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return clean(tok.decode(out[len(ids):]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", default=[])
    ap.add_argument("--interactive", "-i", action="store_true")
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--model", "-m", default=None,
                    help="权重目录（含 model.safetensors 与 tokenizer/）")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    m, tok, dev = load(model_dir=a.model, device=a.device)

    if a.interactive:
        print(f"[ym-Translation] device={dev}  输入空行退出")
        while True:
            try:
                s = input("中> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not s:
                break
            print("英>", translate(s, m, tok, dev, a.max_new))
        return

    text = " ".join(a.text) if a.text else "今天天气怎么样？"
    print("中:", text)
    print("英:", translate(text, m, tok, dev, a.max_new))


if __name__ == "__main__":
    main()
