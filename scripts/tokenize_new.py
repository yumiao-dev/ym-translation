# -*- coding: utf-8 -*-
"""
ym-Translation Stage2: 把新下载的 OPUS 语料 tokenize 成训练用 uint32 分片

【与 tokenize_trans.py 的差异】
1. 数据源：data/new/raw/*.en / *.zh（14 个语料，不是单个 csv）
2. 按语料类别加权重复：口语类（OpenSubtitles/TED）重复更多遍
3. 输出到 data/tokens2/（与语用对 prag_*.bin 同目录），命名 ord_*.bin
4. 保留双向 + 句级/段落级混合

【加权策略】（用户要求：口语优先、高质量优先）
  w=6  OpenSubtitles  口语字幕（治 C/E 类）
  w=5  TED2020/TED2013/NeuLab-TedTalks  口语演讲
  w=4  UNPC           高质人工译（联合国文件）
  w=3  News-Commentary 高质新闻评述
  w=2  MultiUN        正式长句
  w=2  WikiMatrix/wikimedia  句子对齐
  w=1  其余（XLEnt/ALT/QED/Tanzil/PHP/infopankki/WMT-News/tico-19/tldr-pages）
"""
import os, sys, json, time, re, html, glob, random
import numpy as np
from tokenizers import Tokenizer

BASE = "/mnt/workspace/ym-Translation"
RAW = os.path.join(BASE, "data/new/raw")
OUT = os.path.join(BASE, "data/tokens2")
TOK_PATH = glob.glob(os.path.join(BASE, "**/tokenizer.json"), recursive=True)[0]
LOG = "/tmp/tokenize_new.log"

SHARD_BYTES = 192 * 1024 * 1024
BATCH = 2000
MAX_ZH = 512
MAX_EN = 2048
MIN_LEN = 2
MAX_RATIO = 4.0
SEG_SENTENCES = 8
SEG_EVERY_N = 4

# 语料权重（越大越重要）—— 已按 2B token 预算调小
WEIGHTS = {
    "OpenSubtitles_v2016": 3,     # 口语主力
    "TED2020": 3, "TED2013": 3, "NeuLab-TedTalks": 3,
    "UNPC": 2,                    # 高质人工译（17M 对，权重太高会爆）
    "News-Commentary_v16": 3,
    "MultiUN": 1,
    "WikiMatrix": 1, "wikimedia": 1,
}
DEFAULT_W = 1

# 总产出上限（超出即停止），防止产出远超 GPU 配额能训完的量
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2200000000"))   # 2.2B

os.makedirs(OUT, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def clean_zh(t):
    t = html.unescape(t or "")
    t = re.sub(r"\s+", "", t)
    return t.strip()[:MAX_ZH]


def clean_en(t):
    t = html.unescape(t or "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()[:MAX_EN]


class ShardWriter:
    """顺序写入 ord_NN.bin（不覆盖已有的 prag_*.bin）"""
    def __init__(self):
        self.idx = 0
        self.cur = None
        self.cur_n = 0
        self.total = 0
        self._open()

    def _open(self):
        self.path = os.path.join(OUT, f"ord_{self.idx:03d}.bin")
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
            if self.cur_n == 0:
                try:
                    os.remove(self.path)
                except Exception:
                    pass
            else:
                log(f"  -> {os.path.basename(self.path)}  final {self.cur_n/1024**2:.0f}MB")


def load_pairs(name):
    """读 <name>.en / <name>.zh，返回 [(zh, en), ...]"""
    fe = os.path.join(RAW, name + ".en")
    fz = os.path.join(RAW, name + ".zh")
    if not (os.path.exists(fe) and os.path.exists(fz)):
        return []
    pairs = []
    with open(fe, "r", encoding="utf-8", errors="ignore") as a, \
         open(fz, "r", encoding="utf-8", errors="ignore") as b:
        for le, lz in zip(a, b):
            en = clean_en(le)
            zh = clean_zh(lz)
            if len(zh) < MIN_LEN or len(en) < MIN_LEN:
                continue
            r = len(en) / max(len(zh), 1)
            if r > MAX_RATIO or r < 1 / MAX_RATIO:
                continue
            pairs.append((zh, en))
    return pairs


def main():
    tok = Tokenizer.from_file(TOK_PATH)
    L_ZH, L_EN, EOS = "<|zh|>", "<|en|>", "<|eos|>"
    log("=== Stage2 tokenize 新语料 ===")
    log(f"vocab={tok.get_vocab_size()} | RAW={RAW} | OUT={OUT}")

    corpora = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(RAW, "*.en")))
    log(f"发现 {len(corpora)} 个语料: {corpora}")

    writer = ShardWriter()
    buf = []
    stats = {"pairs": 0, "sent": 0, "seg": 0, "skip": 0}
    t0 = time.time()

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

    for name in corpora:
        if writer.total >= MAX_TOKENS:
            log(f"[STOP] 已达上限 {MAX_TOKENS/1e9:.2f}B tokens，跳过剩余语料")
            break
        w = WEIGHTS.get(name, DEFAULT_W)
        pairs = load_pairs(name)
        if not pairs:
            log(f"[SKIP] {name} 无数据")
            continue
        log(f"[LOAD] {name}  w={w}  pairs={len(pairs):,}")
        for rep in range(w):
            if writer.total >= MAX_TOKENS:
                log(f"[STOP] 达上限，{name} 只跑了 {rep}/{w} 遍")
                break
            if rep > 0:
                random.shuffle(pairs)
            seg_zh, seg_en = [], []
            for zh, en in pairs:
                stats["pairs"] += 1
                buf.append(f"{L_ZH} {zh} {L_EN} {en} {EOS}")
                buf.append(f"{L_EN} {en} {L_ZH} {zh} {EOS}")
                stats["sent"] += 2
                seg_zh.append(zh)
                seg_en.append(en)
                if len(seg_zh) >= SEG_SENTENCES:
                    pz = "".join(seg_zh)
                    pe = " ".join(seg_en)
                    buf.append(f"{L_ZH} {pz} {L_EN} {pe} {EOS}")
                    buf.append(f"{L_EN} {pe} {L_ZH} {pz} {EOS}")
                    stats["seg"] += 2
                    seg_zh, seg_en = [], []
                if len(buf) >= BATCH:
                    flush()
                    if writer.total >= MAX_TOKENS:
                        break
            flush()
            el = time.time() - t0
            log(f"  rep{rep+1}/{w} done | total {writer.total/1e9:.3f}B tok | "
                f"{writer.total/el/1000:.0f}K tok/s | {(el)/60:.1f}min")
        del pairs

    flush()
    writer.close()

    log(f"=== DONE: {writer.total/1e9:.3f}B tokens | "
        f"pairs={stats['pairs']:,} sent={stats['sent']:,} seg={stats['seg']:,} ===")
    log(f"磁盘: {OUT}")


if __name__ == "__main__":
    main()
