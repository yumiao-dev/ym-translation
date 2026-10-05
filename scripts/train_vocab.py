# -*- coding: utf-8 -*-
"""
ym-Translation Step1: 训练 32K 中英联合 BPE 词表

设计要点：
- **共享词表**（shared vocab）：中英用同一套 embedding，0.1B 模型参数省一半
- **中文去空格**：WMT 语料中文列带分词空格，必须删掉（业界标准做法）
  * 理由1：BPE 词边界靠空格标记，中文带空格会被切碎，多字词学不出来
  * 理由2：实测无空格版 token 效率高 2.4 倍（10 vs 24 tokens）
  * 理由3：输出自然 —— 带空格训练会让模型生成"表演 前 ，"这种怪异中文
- 特殊 token：<pad> <unk> <bos> <eos> <|zh|> <|en|> <|mask|>

产出：
  ym-Translation/tokenizer/  (tokenizer.json, vocab.json, merges.txt, tokenizer_config.json)
"""
import os, sys, json, time, csv, re, random, html
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC

BASE = "/mnt/workspace/ym-Translation"
RAW = os.path.join(BASE, "data/raw/wmt_zh_en_training_corpus.csv")
OUT = os.path.join(BASE, "tokenizer")
SAMPLE = "/tmp/vocab_sample.txt"
LOG = "/tmp/train_vocab.log"

VOCAB_SIZE = 32000
SAMPLE_PAIRS = 400_000       # 抽多少句对来训词表（32K 词表 40万句足够；太大易 OOM）
SAMPLE_MAX_MB = 80           # 样本文件字节上限（防止 BPE 内存爆炸）
SEED = 42
os.makedirs(OUT, exist_ok=True)

# 特殊 token
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<|zh|>", "<|en|>", "<|mask|>"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_zh(t):
    """中文清洗：去空格（关键）+ 统一空白 + 去 HTML 实体"""
    if not t:
        return ""
    t = html.unescape(t)
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # 去掉所有空白 —— 中文不需要空格分隔
    t = re.sub(r"\s+", "", t)
    return t.strip()


def clean_en(t):
    """英文清洗：保留单词间空格 + 归一化空白 + 去 HTML 实体"""
    if not t:
        return ""
    t = html.unescape(t)
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def sniff_and_iter(path, max_pairs):
    """CSV 逐行读取，返回 (zh, en) 迭代器"""
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(65536)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delim = dialect.delimiter
        except Exception:
            delim = ","
        log(f"delimiter = {repr(delim)}")
        reader = csv.reader(f, delimiter=delim)
        n = 0
        for row in reader:
            if n >= max_pairs:
                break
            if len(row) < 2:
                continue
            # WMT 这个源是 「中文,英文」 两列
            zh, en = row[0], row[1]
            n += 1
            yield zh, en


def main():
    random.seed(SEED)
    log("=== Step1: 训练 32K 中英联合 BPE 词表 ===")
    if not os.path.exists(RAW):
        log(f"!! 语料不存在: {RAW}")
        sys.exit(1)
    log(f"size = {os.path.getsize(RAW)/1e9:.2f} GB")

    # ---------- 1. 抽样写样本文件 ----------
    log(f"抽取 {SAMPLE_PAIRS:,} 句对 (上限 {SAMPLE_MAX_MB}MB)...")
    t0 = time.time()
    n = 0
    max_bytes = SAMPLE_MAX_MB * 1024 * 1024
    with open(SAMPLE, "w", encoding="utf-8") as g:
        wb = 0
        for zh, en in sniff_and_iter(RAW, SAMPLE_PAIRS * 3):
            zh = clean_zh(zh)
            en = clean_en(en)
            if len(zh) < 2 or len(en) < 2:
                continue
            # 每行一条：中文不带空格，英文带空格（各自语言的规范形态）
            s = zh + "\n" + en + "\n"
            g.write(s)
            wb += len(s.encode("utf-8"))
            n += 1
            if n >= SAMPLE_PAIRS or wb >= max_bytes:
                break
    log(f"样本 {n:,} 句对 -> {os.path.getsize(SAMPLE)/1e6:.1f}MB  ({time.time()-t0:.0f}s)")

    # ---------- 2. 训 BPE ----------
    log(f"训练 BPE vocab_size={VOCAB_SIZE} ...")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = NFKC()
    # ByteLevel 预分词：语言无关，中文也能处理
    # use_regex=True（默认 GPT-2 切分）：按空格/标点/数字边界切分，大幅降低内存
    #   注意：use_regex=False 会把整段当一个长串，内存爆炸（之前 OOM 的元凶）
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIALS,
        show_progress=True,
        min_frequency=5,             # 调高：降低候选 pair 数，防 OOM
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        limit_alphabet=6000,         # 限制初始字母表大小，内存友好
    )
    t1 = time.time()
    tok.train([SAMPLE], trainer)
    log(f"BPE 训练完成 ({time.time()-t1:.0f}s)  vocab={tok.get_vocab_size()}")

    # ---------- 3. 保存 ----------
    tok.save(os.path.join(OUT, "tokenizer.json"))
    tok.model.save(os.path.join(OUT, "vocab.json"), os.path.join(OUT, "merges.txt"))
    with open(os.path.join(OUT, "tokenizer_config.json"), "w", encoding="utf-8") as g:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 8192,
            "unk_token": "<unk>", "bos_token": "<bos>", "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": ["<|zh|>", "<|en|>", "<|mask|>"],
        }, g, ensure_ascii=False, indent=2)
    log(f"保存到 {OUT}")

    # ---------- 4. 验证 ----------
    tests = [
        "表演前，她紧张得浑身颤抖不已。",
        "She was so nervous before the performance that she was shaking like a jelly.",
        "<|zh|>你好世界<|en|>Hello world<|eos|>",
        "联合国安理会常任理事国一致通过了这项决议。",
        "The quick brown fox jumps over the lazy dog.",
    ]
    with open(os.path.join(OUT, "verify.txt"), "w", encoding="utf-8") as g:
        for t in tests:
            e = tok.encode(t)
            line = f"{t!r}\n  n={len(e.ids)}  ids={e.ids[:24]}\n  toks={e.tokens[:16]}"
            log(line)
            g.write(line + "\n")

    # token 效率统计
    log("=== 完成 ===")
    log(f"最终词表大小 = {tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
