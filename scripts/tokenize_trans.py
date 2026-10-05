# -*- coding: utf-8 -*-
"""
ym-Translation Step2: 把 WMT 中英平行语料 tokenize 成训练用 uint32 二进制分片

设计（面向端侧翻译模型）：
  样本格式（双向 + 句级/段落级混合）：
    句级：  <|zh|> 中文句 <|en|> English sentence <|eos|>
    段落：  <|zh|> [整段中文，由N个连续句拼接] <|en|> [对应整段英文] <|eos|>
  样本之间用 <|eos|> 分隔，首尾相连成流。

  关键处理：
    1. 中文去空格（去掉分词空格）
    2. HTML 实体解码
    3. 长度过滤（源/目标长度比、最大长度）
    4. 段落级：按 MAX_SEG 句为一组拼接，制造长上下文样本

产出：
  data/tokens/part_NN.bin (uint32, 每 256MB 一个)
  data/tokens/meta.json
  data/tokens/done.txt
"""
import os, sys, json, time, csv, re, html, glob
import numpy as np
from tokenizers import Tokenizer

BASE = "/mnt/workspace/ym-Translation"
RAW = os.path.join(BASE, "data/raw/wmt_zh_en_training_corpus.csv")
OUT = os.path.join(BASE, "data/tokens")
TOK = os.path.join(BASE, "tokenizer/tokenizer.json")
LOG = "/tmp/trans_tokenize.log"

TARGET_TOK = int(os.environ.get("TARGET_TOK", "2000000000"))  # 2B
SHARD_BYTES = 256 * 1024 * 1024
BATCH = 2000

# 长度限制
MAX_ZH = 512
MAX_EN = 2048
MIN_LEN = 2
MAX_RATIO = 4.0          # 源/目标字符长度比上限

# 段落级样本：每 N 句拼一段（制造 4-8K 上下文样本）
SEG_SENTENCES = 8
SEG_EVERY_N = 5          # 每 5 个句级样本，产 1 个段落样本

os.makedirs(OUT, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_zh(t):
    t = html.unescape(t or "")
    t = re.sub(r"\s+", "", t)      # 中文去所有空格
    return t.strip()


def clean_en(t):
    t = html.unescape(t or "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()[:MAX_EN]


class ShardWriter:
    def __init__(self):
        self.idx = len(glob.glob(os.path.join(OUT, "part_*.bin")))
        self.cur = None
        self.cur_n = 0
        self.total = 0
        self._open()

    def _open(self):
        self.path = os.path.join(OUT, f"part_{self.idx:02d}.bin")
        self.cur = open(self.path, "wb")
        self.cur_n = 0

    def write(self, arr):
        self.cur.write(arr.tobytes())
        self.cur_n += arr.nbytes
        self.total += len(arr)
        if self.cur_n >= SHARD_BYTES:
            self.cur.close()
            log(f"  -> {os.path.basename(self.path)}  {self.cur_n/1024**3:.2f}GB")
            self.idx += 1
            self._open()

    def close(self):
        if self.cur:
            self.cur.close()
            log(f"  -> {os.path.basename(self.path)}  final {self.cur_n/1024**2:.0f}MB")


def main():
    tok = Tokenizer.from_file(TOK)
    log("=== Step2: tokenize 中英平行语料 ===")
    log(f"target = {TARGET_TOK/1e9:.2f}B tokens | vocab = {tok.get_vocab_size()}")

    L_ZH, L_EN, EOS = "<|zh|>", "<|en|>", "<|eos|>"

    writer = ShardWriter()
    buf = []
    stats = {"pairs": 0, "sent_ok": 0, "seg_ok": 0, "skip": 0}

    def flush():
        nonlocal buf
        if not buf:
            return
        encs = tok.encode_batch(buf)
        ids = []
        for e in encs:
            ids.extend(e.ids)
        if ids:
            writer.write(np.array(ids, dtype=np.uint32))
        buf = []

    # 段落缓冲
    seg_zh, seg_en = [], []

    t0 = time.time()
    with open(RAW, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if writer.total >= TARGET_TOK:
                break
            if len(row) < 2:
                continue
            zh = clean_zh(row[0])[:MAX_ZH]
            en = clean_en(row[1])
            if len(zh) < MIN_LEN or len(en) < MIN_LEN:
                stats["skip"] += 1
                continue
            # 长度比过滤
            r = len(en) / max(len(zh), 1)
            if r > MAX_RATIO or r < 1 / MAX_RATIO:
                stats["skip"] += 1
                continue

            stats["pairs"] += 1

            # --- 句级双向 ---
            buf.append(f"{L_ZH} {zh} {L_EN} {en} {EOS}")
            buf.append(f"{L_EN} {en} {L_ZH} {zh} {EOS}")
            stats["sent_ok"] += 2

            # --- 段落级 ---
            seg_zh.append(zh)
            seg_en.append(en)
            if len(seg_zh) >= SEG_SENTENCES:
                pz = "".join(seg_zh)
                pe = " ".join(seg_en)
                buf.append(f"{L_ZH} {pz} {L_EN} {pe} {EOS}")
                buf.append(f"{L_EN} {pe} {L_ZH} {pz} {EOS}")
                stats["seg_ok"] += 2
                seg_zh, seg_en = [], []

            if len(buf) >= BATCH:
                flush()
                if stats["pairs"] % 100000 == 0:
                    el = time.time() - t0
                    log(f"  {stats['pairs']:,} pairs  {writer.total/1e9:.3f}B tok  "
                        f"{writer.total/el/1000:.0f}K tok/s")

    flush()
    writer.close()

    meta = {
        "target_tokens": TARGET_TOK,
        "total_tokens": writer.total,
        "pairs": stats["pairs"],
        "sent_samples": stats["sent_ok"],
        "seg_samples": stats["seg_ok"],
        "skipped": stats["skip"],
        "shards": writer.idx + 1,
        "vocab_size": tok.get_vocab_size(),
        "markers": {"zh": L_ZH, "en": L_EN, "eos": EOS},
    }
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as g:
        json.dump(meta, g, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "done.txt"), "w") as g:
        g.write(f"done total={writer.total}\n")
    log(f"=== DONE: {writer.total/1e9:.3f}B tokens, {stats['pairs']:,} pairs ===")


if __name__ == "__main__":
    main()
