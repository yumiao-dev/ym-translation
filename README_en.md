# ym-Translation

**[中文](README.md) | English**

> A **54.4M-parameter** bidirectional Chinese↔English translation model, trained **from scratch**,
> targeting **fully offline** inference on Android devices.
> **Trained on a single AMD Instinct MI300A — 20 h 40 m of pure training, 8.3B tokens.**

[![params](https://img.shields.io/badge/params-54.4M-blue)]()
[![tokens](https://img.shields.io/badge/trained-8.3B_tokens-green)]()
[![bleu](https://img.shields.io/badge/FLORES--200_BLEU-17.50-orange)]()
[![chrf](https://img.shields.io/badge/chrF%2B%2B-43.98-blueviolet)]()
[![context](https://img.shields.io/badge/context-4K_(extrapolatable_8K)-orange)]()
[![hardware](https://img.shields.io/badge/hardware-1%C3%97_AMD_MI300A-red)]()
[![time](https://img.shields.io/badge/train-20h40m-purple)]()
[![status](https://img.shields.io/badge/status-work_in_progress-yellow)]()
[![hf](https://img.shields.io/badge/%F0%9F%A4%97_weights-Hugging_Face-yellow)](https://huggingface.co/yumiao-dev/ym-translation)

---

## Model weights

Weights are hosted on **Hugging Face**. This repo contains code and docs only —
**no weight files**:

| | |
|---|---|
| 🤗 Repo | [yumiao-dev/ym-translation](https://huggingface.co/yumiao-dev/ym-translation) |
| Files | `model.safetensors` (208 MB, FP32, **Stage 3**) + config + tokenizer |
| One-line download | `huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights` |

---

## What is this

A **from-scratch** (not distilled, not fine-tuned) bidirectional Chinese↔English
translation model. The goal is a translator that runs **without network access,
without API calls, entirely on a phone NPU**.

Unlike the mainstream approach (distilling a large model), this project goes with
**from-scratch pretraining + a self-trained 32K vocabulary**, to squeeze the maximum
usable capacity out of a tiny parameter budget.

**Current status: Stage 3.** FLORES-200 devtest zh→en BLEU **17.50** / chrF++ **43.98**,
reaching **86.3% of OPUS-MT's chrF++**. Everyday sentences already match the baseline;
conversational pragmatics is still the weak spot.
**See the [Evaluation](#evaluation) section — we record the failures honestly.**

---

## Positioning & comparison

> Everything below is **our own head-to-head testing**, not a third-party ranking.
> Test setup and raw data live in [`eval/`](eval).

### Where we claim to stand (qualified wording)

| Claim | Basis |
|---|---|
| We **could not find a smaller open-source bidirectional zh↔en** model | 54.4M, one model for both directions |
| **From scratch**: random init, trained in stages to 8.3B tokens | not distilled, not fine-tuned |
| Highest FLORES-200 zh→en BLEU **within our 50–60M comparison set** | 17.50 |
| Parameter efficiency **0.322 BLEU per million params** | vs OPUS-MT 0.297, m2m100 0.046 |
| **One of the few** decoder-only models reaching usable quality at 54M | mainstream MT is encoder-decoder |
| Apache-2.0: weights + code + training scripts all open | no commercial restriction |

### FLORES-200 devtest zh→en · head-to-head (1012 sentences)

| Model | Params | Architecture | Direction | BLEU | chrF |
|---|---|---|---|---|---|
| OPUS-MT (Helsinki) | 77.9M ×2 | Encoder-Decoder | one-way zh→en | **23.16** | **53.60** |
| m2m100-418M (Meta) | 418M | Encoder-Decoder | 100 languages | 19.14 | 49.99 |
| **★ ym-translation Stage 3** | **54.4M** | Decoder-only | **bidirectional** | 17.50 | 46.46 |
| ym-translation Stage 2 | 54.4M | Decoder-only | bidirectional | 11.80 | 41.77 |

- `chrF` here is sacrebleu's `chrF`; ym / OPUS `chrF++` are **43.98 / 50.97**.
- The ym and OPUS rows are **re-computable offline** from [`eval/`](eval);
  the m2m100 row was measured in **our own sandbox** (CPU, default transformers decoding)
  and is not shipped in `eval/`.

### Parameter efficiency

| Model | BLEU per million params |
|---|---|
| **ym-translation Stage 3** | **0.322** |
| OPUS-MT (one direction) | 0.297 |
| m2m100-418M | 0.046 |

### The real cost of bidirectional translation

| Setup | Total params for both directions |
|---|---|
| OPUS-MT | 77.9M (zh→en) + 77.9M (en→zh) = **155.8M** |
| **ym-translation** | **54.4M** (one model, both directions) |

For zh↔en, ym uses **~35%** of the parameters and keeps **75.6%** of OPUS-MT's one-way BLEU.

### Quantized size (**projected**)

> int8 / int4 quantization is **not implemented yet** (roadmap P3); the numbers below
> are estimates derived from the parameter count.

| Model | FP32 | int8 | int4 |
|---|---|---|---|
| **ym-translation** | **208 MB** | ~54 MB | ~27 MB |
| OPUS-MT (both directions) | ~0.62 GB | ~0.16 GB | ~78 MB |
| m2m100-418M | 1.94 GB | ~0.48 GB | ~0.24 GB |

### Inference speed

| Platform | ym-translation | OPUS-MT (beam=4) | m2m100-418M |
|---|---|---|---|
| MI300X GPU (single sentence, bf16 / greedy) | **184 ms/sent** | 34 ms/sent | — |
| Our sandbox CPU (FP32 / greedy) | **0.99 s/sent** | ~1.5 s/sent | ~1.3 s/sent |

The GPU row uses the same setup as [`eval/`](eval); the CPU row was measured in
**our own sandbox** (m2m100 took ~22 min for all 1012 sentences) and is provided
as an on-device reference.

---

## Release artifacts

| File | Description |
|---|---|
| [`modeling_ym.py`](modeling_ym.py) | Model definition, loads the weights standalone (torch only) |
| [`infer.py`](infer.py) | Ready-to-run inference script (CPU / GPU) |
| [`config.json`](config.json) | Architecture params + training metadata |
| [`tokenizer/`](tokenizer) | Self-trained 32K joint BPE vocabulary |
| [`eval/`](eval) | FLORES-200 devtest raw data + three sacrebleu score files (**re-computable offline**) |
| [`FLORES-200_评测报告.md`](FLORES-200_评测报告.md) | Full evaluation report |
| [`scripts/`](scripts) | Training / tokenization / data-building scripts |
| [`docs/`](docs) | Environment notes, dataset inventory, training reports |
| [`eval_results/`](eval_results) | Raw sentence-by-sentence human comparison logs |

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

Decoder-only Transformer, 54.4M parameters (tied embedding counted once).

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

> The exported `state_dict` has 102 keys; `emb.weight` and `head.weight`
> **share one tensor**, so only `emb.weight` is stored.
> Actual parameter count: **54,406,912**.

---

## Training setup

### Hardware

| Item | Spec |
|---|---|
| Platform | ModelScope Code Workspace (DSW-AMD) |
| Accelerator | **AMD Instinct MI300A** (APU unified memory) |
| Memory | 191.7 GB |
| GPU utilization | 96–100% (saturated the whole time) |

### Wall-clock time

| Stage | Steps | Tokens | GPU time |
|---|---|---|---|
| Stage 1 (base training) | 0 → 3,815 | 2.00B | 4h59m |
| Stage 2 (continued, conversational) | 3,815 → 8,010 | 2.20B | 5h29m |
| Continued | 8,010 → 13,506 | 2.88B | 7h10m |
| **Stage 3** | 13,506 → 15,830 | 1.22B | 3h02m |
| **Total** | **0 → 15,830** | **8.30B** | **20h40m** |

**4.699 s/step** throughout (global batch = 524,288 tokens/step), cross-checked against
both the checkpoint timestamps and the training log — zero jitter.
Stage 3 throughput ≈ **760K tok/s**.

### Hyperparameters

| Item | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| LR peak → floor | 1e-3 → 1e-4 | 3e-4 → 3e-5 | 3e-4 → 3e-5 |
| warmup | 100 | 60 | 60 |
| micro batch × grad accum | 16 × 8 | 8 × 16 | 8 × 16 |
| global batch | 524,288 tok/step | same | same |
| optimizer | AdamW (0.9, 0.95), wd=0.1, fused | same | same |
| precision | bf16 autocast | same | same |

> Training ran in bf16 autocast; the **released weights are FP32** (208 MB).
> For on-device deployment you'll want to convert to bf16/int8 yourself —
> that halves the size or better.

---

## Data

| Item | Value |
|---|---|
| Training tokens | 8.3B (15,830 steps × 524,288) |
| Vocabulary | 32,000 (self-trained) |

| Stage | Steps | Tokens | Source |
|---|---|---|---|
| Stage 1 | 0 → 3,815 | 2.00B | ModelScope `iic/WMT-zh-en` (6.36 GB, ~25M pairs) |
| Stage 2 | 3,815 → 8,010 | 2.20B | 14 OPUS corpora (8.0 GB) + **pragmatic equivalence pairs** |
| Continued | 8,010 → 13,506 | 2.88B | Extended OPUS corpora |
| Stage 3 | 13,506 → 15,830 | 1.22B | Extended OPUS + augmented pragmatic pairs |

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

### FLORES-200 devtest · Chinese→English (1012 sentences)

| Metric | **ym (Stage 3)** | ym (Stage 2) | OPUS-MT | ym/OPUS |
|---|---|---|---|---|
| **BLEU (13a)** | **17.50** | 11.80 | **23.16** | 75.6% |
| BLEU (zh) | 17.44 | 9.76 | 23.07 | 75.6% |
| **chrF** | **46.46** | 41.77 | **53.60** | 86.7% |
| **chrF++** | **43.98** | 39.27 | **50.97** | 86.3% |
| TER | 76.05 | 103.30 | 65.74 | — |
| Avg output words | **21.49** | 27.35 | 21.79 | — |
| Empty outputs | 0 | 0 | 0 | — |

Reference translations average **21.64 words**. All data and scores live in
[`eval/`](eval) and **can be re-computed offline with sacrebleu**.

> **How to read this**: chrF++ at **86.3%** of the baseline means the model
> **gets the content right**; BLEU at **75.6%** means it **doesn't sound natural yet**.
> The gap between the two is the classic signature of a tiny model.
>
> ym has only **70%** of OPUS-MT's parameters (54.4M vs 77.9M) and is a **single
> bidirectional model**, while OPUS needs one model per direction. ym also uses
> **greedy decoding** against beam=4 — which puts ym at a disadvantage.

### Stage 2 → Stage 3: what doubling the tokens bought

| Metric | Stage 2 (4.2B) | Stage 3 (8.3B) | Change |
|---|---|---|---|
| BLEU (13a) | 11.80 | **17.50** | **+48.3%** |
| chrF++ | 39.27 | **43.98** | +12.0% |
| TER | 103.30 | **76.05** | **−26.4%** |
| Avg output words | 27.35 | **21.49** | −21.4% |
| Generation speed | 1099 ms/sent | **184 ms/sent** | **6.0×** |

Stage 2 emitted 27.35 words where the reference has 21.64, with TER at 103.3
(above 100 means it edits more than simply copying the reference).
Stage 3 converges to 21.49 words.

> **Output length regression + 6x speedup = one root cause: repetition is largely suppressed.**
> This isn't a capability suddenly appearing — it's long-tail defects being systematically fixed.

### Human comparison · 47 short sentences vs OPUS-MT

| | Stage 2 | Stage 3 |
|---|---|---|
| ym wins | 0 | **2** |
| OPUS wins | 29 | **17** |
| ties | 12 | **27** |
| both wrong | 4 | **0** |

> The gain isn't one category getting stronger — it's **"both wrong" and
> "OPUS-exclusive wins" turning into ties across the board**.
> Sentence-by-sentence logs in [`eval_results/`](eval_results).

### Long text

In the Stage 2 round, across 13 long-text cases: ym 0 wins / MT 8 wins / both wrong 4.
Typical failures included an **infinite loop** (`he was a new engineer` repeated 68 times)
and **translating only the second half**. Repetition improved substantially in Stage 3,
but long text remains a weak spot.

### Generation config (fairness note)

| Item | ym-Translation | OPUS-MT |
|---|---|---|
| Decoding | **greedy (argmax)** | beam search (beam=4) |
| max tokens | 192 | 256 |
| Precision | bfloat16 | fp32 |
| Batch | 1 | 32 |
| Speed | **184 ms/sent** | 34 ms/sent |

---

## Known issues

| Issue | Status |
|---|---|
| Phrase-level repetition (`medal medals`) | ⚠️ improved, still present |
| Long-text repetition / truncation can cut mid-sentence | ⚠️ improved |
| Conversational responses, pragmatics | ❌ biggest weakness |
| Rare-word spelling (`exports` ← `exits`) | ❌ unsolved |

### Roadmap

| Priority | Action |
|---|---|
| **P0** | beam search (beam=4) + repetition penalty |
| **P1** | add conversational data (OpenSubtitles) |
| **P1** | raise pragmatic-pair REPEAT to 5000+ |
| **P2** | length penalty + over-long segmentation |
| **P3** | int8/int4 quantization + ONNX export |

---

## Quick start

### Run inference with the Stage 3 weights

```bash
pip install -r requirements.txt
huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights

python3 infer.py --model ym-weights "今天天气怎么样？"
# 中: 今天天气怎么样？
# 英: How's the weather today?

python3 infer.py --model ym-weights -i          # interactive mode
```

### Retrain from scratch

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

### Re-compute the evaluation scores (no GPU needed)

See [`eval/README.md`](eval/README.md) — the repo ships every hypothesis file,
so sacrebleu reproduces the scores directly.

---

## Lessons learned (worth a read)

1. **The LR schedule must be based on relative steps** — otherwise LR collapses to
   the floor immediately on continued training.
2. **`TARGET_TOKENS` is cumulative**, not the amount to add.
3. **The data glob `part_*.bin` did not match `ord_*.bin`** — nearly wasted 2.2B tokens.
4. **Throughput unit trap** — the tok/s in the log includes historical tokens in
   its numerator; the real speed is the per-step rate.
5. **Chinese must have spaces removed** — otherwise BPE shatters multi-character words
   and token efficiency drops by 2.4x.

Details in [`docs/环境备忘.md`](docs/环境备忘.md) and [`docs/Stage2训练完成报告.md`](docs/Stage2训练完成报告.md).

---

## Citation

```bibtex
@misc{ymtranslation2026,
  title  = {ym-Translation: A 54M From-Scratch Chinese-English Translation Model},
  year   = {2026},
  note   = {Trained on 8.3B tokens with pragmatic-equivalence pairs},
  url    = {https://github.com/yumiao-dev/ym-translation}
}
```

## License

Apache-2.0

## Acknowledgements

- Corpora: [OPUS](https://object.pouta.csc.fi), ModelScope `iic/WMT-zh-en`
- Baseline: [Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- Platform: ModelScope Code Workspace / AMD Instinct MI300A
