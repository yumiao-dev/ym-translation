# ym-Translation

**[中文](README.md) | English**

> A **54.4M-parameter** bidirectional Chinese↔English translation model, trained **from scratch**,
> targeting **fully offline** inference on Android devices.
> **Trained on a single AMD Instinct MI300A — about 8.2 hours of pure training, 4.2B tokens.**

[![params](https://img.shields.io/badge/params-54.4M-blue)]()
[![tokens](https://img.shields.io/badge/trained-4.2B_tokens-green)]()
[![context](https://img.shields.io/badge/context-4K_(extrapolatable_8K)-orange)]()
[![hardware](https://img.shields.io/badge/hardware-1%C3%97_AMD_MI300A-red)]()
[![time](https://img.shields.io/badge/train-~8.2h-purple)]()
[![status](https://img.shields.io/badge/status-work_in_progress-yellow)]()
[![hf](https://img.shields.io/badge/%F0%9F%A4%97_weights-Hugging_Face-yellow)](https://huggingface.co/yumiao-dev/ym-translation)

---

## Model weights

Weights are hosted on **Hugging Face**. This repo contains code and docs only —
**no weight files**:

| | |
|---|---|
| 🤗 Repo | [yumiao-dev/ym-translation](https://huggingface.co/yumiao-dev/ym-translation) |
| Files | `model.safetensors` (135 MB, FP16) + config + tokenizer |
| One-line download | `huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights` |

---

## What is this

A **from-scratch** (not distilled, not fine-tuned) bidirectional Chinese↔English
translation model. The goal is a translator that runs **without network access,
without API calls, entirely on a phone NPU**.

Unlike the mainstream approach (distilling a large model), this project goes with
**from-scratch pretraining + a self-trained 32K vocabulary**, to squeeze the maximum
usable capacity out of a tiny parameter budget.

**Current status**: work in progress. Everyday sentences are on par with OPUS-MT,
but conversational response particles and long-text stability are still clearly behind.
**See the [Evaluation](#evaluation) section — we record the failures honestly.**

---

## Core design

### Why a self-trained 32K vocabulary

Reusing a general-purpose 151,643-token vocabulary from a large model would make the
**embedding layer alone 116M parameters** — already over the entire 0.1B budget.

Switching to a self-trained 32,000 joint BPE vocabulary + tied embeddings:

| Option | Embedding params |
|---|---|
| Qwen 151,643 vocab | 116M ❌ |
| **Self-trained 32,000 vocab** | **16.4M** ✅ |

The ~100M saved goes entirely into network depth.
**This was the decisive step that made a 0.1B model feasible.**

### Pragmatic equivalence pairs (the core idea here)

The mistake in `我知道了。→ I know.` is one that **even Helsinki-NLP's OPUS-MT makes**.
That tells us pragmatic mapping is a **data-scarcity problem, not a model-size problem**.

So 1,602 **pragmatic equivalence pairs** were hand-built — pairs that don't match
literally but do match pragmatically:

| Category | Example |
|---|---|
| Response particles | `我知道了。 → I see.` (not "I know") |
| Discourse markers | `差不多得了。 → That's enough.` |
| Polite modesty | `哪里哪里。 → You flatter me.` |
| Interjections | `哎呀 → Oh` (not "Gee") |
| Aspect markers | `我吃了。 → I ate it.` |

→ see [`scripts/build_pragmatic_pairs.py`](scripts/build_pragmatic_pairs.py)

---

## Architecture

Decoder-only Transformer, 54.4M parameters.

```python
HIDDEN    = 512
LAYERS    = 10
HEADS     = 8        # query heads
KVHEADS   = 2        # KV heads (GQA, 4x smaller KV cache)
HEAD_DIM  = 64
FFN       = 2048     # SwiGLU
VOCAB     = 32000
CTX       = 4096     # RoPE base=500000, extrapolates to 8K
```

| Component | Choice | Why |
|---|---|---|
| Normalization | RMSNorm | more stable at this scale |
| Position encoding | RoPE base=500000 | large base extrapolates better to long context |
| Attention | GQA (8Q/2KV) | 4x smaller KV cache → critical for on-device memory |
| QK-Norm | enabled | stabilizes training at high LR (peak 1e-3) |
| FFN | SwiGLU | beats GELU MLP at equal params |
| Embeddings | tied | saves 16.4M params |
| Loss | **Chunked CE** | avoids OOM from the `[B,T,32000]` logits matrix |

---

## Training setup

### Hardware

| Item | Spec |
|---|---|
| Platform | ModelScope Code Workspace (DSW-AMD) |
| Accelerator | **AMD Instinct MI300A** (APU unified memory) |
| Memory | 191.7 GB |
| ROCm | 7.2.53211 |
| GPU utilization | 96–100% (saturated the whole time) |

### Wall-clock time

| Stage | Duration | Tokens |
|---|---|---|
| Stage 1 (base training) | ~2.7 h | 2.0B |
| Stage 2 (continued training, conversational) | **5.47 h** | 2.2B (4.2B cumulative) |
| **Training total** | **~8.2 h** | **4.2B** |

Stage 2 measured **213K tok/s**, **4.7 s/step** (zero jitter throughout),
final loss **0.7393**.

### Hyperparameters

| Item | Stage 1 | Stage 2 |
|---|---|---|
| LR peak → floor | 1e-3 → 1e-4 | 3e-4 → 3e-5 |
| warmup | 100 | 60 |
| micro batch | 16 | 8 |
| grad accum | 8 | 16 |
| global batch | 524,288 tok/step | 524,288 tok/step |
| optimizer | AdamW (0.9, 0.95), wd=0.1, fused | same |
| precision | bf16 autocast | same |
| steps | 3,815 | 8,010 |

---

## Data

| Item | Value |
|---|---|
| Raw corpus | 15.9 GB |
| Training tokens | 4.2B |
| Vocabulary | 32,000 (self-trained) |

**Stage 1**: ModelScope `iic/WMT-zh-en` (6.36 GB, ~25M sentence pairs)

**Stage 2**: 14 OPUS corpora (8.0 GB) + pragmatic equivalence pairs (39.9M tokens)

→ full list in [`docs/数据集清单_OPUS.md`](docs/数据集清单_OPUS.md)

Conversational (OpenSubtitles/TED) and news (News-Commentary) data are **weighted x3**.

**Sample format**:
```
sentence: <|zh|> 中文 <|en|> English <|eos|>
paragraph: <|zh|> [8 sentences joined] <|en|> [matching English paragraph] <|eos|>
```

Paragraph-level samples are used to train 4–8K long context —
**machine translation goes sentence by sentence; we translate whole paragraphs.**

---

## Evaluation

### Chinese→English · 47 short sentences vs OPUS-MT

| Category | n | ym wins | MT wins | tie | both wrong |
|---|---|---|---|---|---|
| A sentence-final particles | 6 | 0 | 3 | 3 | 0 |
| B aspect / tense | 5 | 0 | 3 | 2 | 0 |
| C conversational responses | 12 | 0 | 10 | 0 | 2 |
| D interjections | 5 | 0 | 1 | 4 | 0 |
| E discourse markers | 10 | 0 | 8 | 1 | 1 |
| F polite modesty | 6 | 0 | 4 | 1 | 1 |
| **G control group** | 3 | 0 | 0 | **3** | 0 |
| **Total** | **47** | **0** | **29 (62%)** | **14 (30%)** | **4 (9%)** |

> **Verdict: OPUS-MT wins outright.** But ⚠️ **the control group is a perfect 3:3 tie** —
> everyday sentences already match the MT baseline; the gap is only in
> conversational / pragmatic speech.

### Long text · 13 cases

ym 0 wins / MT 8 wins / both wrong 4.

Typical failures (recorded honestly):
- Paragraph-narrative: **infinite loop**, `he was a new engineer` repeated 68 times
- Consistency-terminology: **only the second half translated**, first 60 chars dropped
- Long sentence-technical: **both wrong** (we rendered 4096 as "four-thousand-ninety-six"; MT produced `400096 medals`)

→ full sentence-by-sentence comparison in [`eval_results/ym_vs_机翻_完整对比.txt`](eval_results/ym_vs_机翻_完整对比.txt)

### Speed

| Scenario | ym-Translation | OPUS-MT (beam=4) |
|---|---|---|
| Short sentence (the common on-device case) | **0.19–0.29 s/sentence** | 0.20–0.99 s/sentence |

---

## Known issues

| Issue | Status |
|---|---|
| Conversational responses (category C, 0:10) | ❌ biggest weakness |
| English→Chinese collapse on isolated particles | ❌ corpus leakage + looping |
| Long-text looping / truncation | ❌ needs beam search |

### Roadmap

| Priority | Action |
|---|---|
| **P0** | add beam search + repetition penalty (fix looping) |
| **P1** | add conversational data (OpenSubtitles) |
| **P1** | raise pragmatic-pair REPEAT to 5000+ (**could overtake MT**) |
| **P2** | length penalty + over-long segmentation |
| **P3** | int8/int4 quantization + ONNX export |

---

## Quick start

```bash
pip install -r requirements.txt

# ── Stage 1 ──
python3 scripts/train_vocab.py                        # train the tokenizer
python3 scripts/tokenize_trans.py                     # tokenize
TARGET_TOKENS=2000000000 python3 scripts/train_trans.py

# ── Stage 2 ──
bash scripts/dl_new_corpora.sh                        # download OPUS corpora
python3 scripts/build_pragmatic_pairs.py              # build pragmatic pairs
python3 scripts/tokenize_new.py
bash scripts/start_stage2.sh

# ── Evaluation ──
python3 scripts/eval_v3.py    --ckpt output/ckpt2/latest.pt
python3 scripts/eval_long.py  --ckpt output/ckpt2/latest.pt
```

---

## Lessons learned (worth a read)

1. **The LR schedule must be based on relative steps** — otherwise LR collapses to
   the floor immediately on continued training.
2. **`TARGET_TOKENS` is cumulative**, not the amount to add.
3. **The data glob `part_*.bin` did not match `ord_*.bin`** — nearly wasted 2.2B tokens.
4. **Throughput unit trap** — the `213K tok/s` in the log includes historical tokens in
   its numerator; the real speed is the per-step rate.
5. **Chinese must have spaces removed** — otherwise BPE shatters multi-character words
   and token efficiency drops by 2.4x.

Details in [`docs/环境备忘.md`](docs/环境备忘.md) and [`docs/Stage2训练完成报告.md`](docs/Stage2训练完成报告.md).

---

## License

Apache-2.0

## Acknowledgements

- Corpora: [OPUS](https://object.pouta.csc.fi), ModelScope `iic/WMT-zh-en`
- Baseline: [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- Platform: ModelScope Code Workspace / AMD Instinct MI300A