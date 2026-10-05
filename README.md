# ym-Translation

**中文 | [English](README_en.md)**

> 从零预训练的 **54.4M 参数**中英双向翻译小模型，目标是在安卓端侧**离线**运行。
> **单卡 AMD Instinct MI300A，纯训练约 8.2 小时，4.2B tokens。**

[![params](https://img.shields.io/badge/params-54.4M-blue)]()
[![tokens](https://img.shields.io/badge/trained-4.2B_tokens-green)]()
[![context](https://img.shields.io/badge/context-4K_(extrapolatable_8K)-orange)]()
[![hardware](https://img.shields.io/badge/hardware-1%C3%97_AMD_MI300A-red)]()
[![time](https://img.shields.io/badge/train-~8.2h-purple)]()
[![status](https://img.shields.io/badge/status-work_in_progress-yellow)]()
[![hf](https://img.shields.io/badge/%F0%9F%A4%97_weights-Hugging_Face-yellow)](https://huggingface.co/yumiao-dev/ym-translation)

---

## 模型权重

权重托管在 **Hugging Face**，本仓库只放代码与文档、**不含权重文件**：

| | |
|---|---|
| 🤗 仓库 | [yumiao-dev/ym-translation](https://huggingface.co/yumiao-dev/ym-translation) |
| 文件 | `model.safetensors`（135 MB, FP16）+ config + tokenizer |
| 一条命令下载 | `huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights` |

> 国内网络可先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 这是什么

一个**从零预训练**（非蒸馏、非微调）的中英双向翻译模型。作者的目标是做一个
**不联网、不调 API、完全跑在手机 NPU 上**的翻译模型。

区别于主流做法（拿大模型蒸馏），本项目选择 **从零预训练 + 自训 32K 词表**，
以在极小参数预算下榨出最大有效容量。

**当前状态**：实验进行中。日常句子已追平 OPUS-MT 机翻，
但口语应答语与长文本稳定性仍明显落后。**详见[评估结果](#评估结果)一节——我们如实记录失败。**

---

## 核心设计

### 为什么自训 32K 词表

沿用通用大模型的 151,643 词表，**光 embedding 一层就占 116M 参数**，
直接超出整个 0.1B 预算。

换成自训 32,000 联合 BPE 词表 + tied embedding 后：

| 方案 | Embedding 参数 |
|---|---|
| Qwen 151,643 词表 | 116M ❌ |
| **自训 32,000 词表** | **16.4M** ✅ |

省下的 ~100M 全部留给网络深度。**这是让 0.1B 模型跑得动的决定性一步。**

### 语用等价对（本项目核心创新）

`我知道了。→ I know.` 这个错，**连 Helsinki-NLP 的 OPUS-MT 都犯**。
这说明语用映射是**数据稀缺问题**，不是模型大小问题。

于是作者手工构造了 1,602 条**语用等价对**——字面不对应、但语用功能对应的翻译对：

| 类别 | 示例 |
|---|---|
| 应答语 | `我知道了。 → I see.`（而非 I know） |
| 话语标记 | `差不多得了。 → That's enough.` |
| 委婉敬语 | `哪里哪里。 → You flatter me.` |
| 感叹词 | `哎呀 → Oh`（而非 Gee） |
| 体标记 | `我吃了。 → I ate it.` |

→ 见 [`scripts/build_pragmatic_pairs.py`](scripts/build_pragmatic_pairs.py)

---

## 模型架构

Decoder-only Transformer，54.4M 参数。

```python
HIDDEN    = 512
LAYERS    = 10
HEADS     = 8        # Query 头
KVHEADS   = 2        # KV 头（GQA，KV cache 省 4 倍）
HEAD_DIM  = 64
FFN       = 2048     # SwiGLU
VOCAB     = 32000
CTX       = 4096     # RoPE base=500000，可外推 8K
```

| 组件 | 选择 | 理由 |
|---|---|---|
| 归一化 | RMSNorm | 小模型上更稳 |
| 位置编码 | RoPE base=500000 | 大 base 利于长上下文外推 |
| 注意力 | GQA (8Q/2KV) | KV cache 缩小 4 倍 → 端侧内存关键 |
| QK-Norm | 启用 | 稳定大 LR 训练（峰值 1e-3） |
| FFN | SwiGLU | 同参数优于 GELU MLP |
| 词嵌入 | Tied | 省 16.4M 参数 |
| 损失 | **Chunked CE** | 解决 `[B,T,32000]` 巨矩阵 OOM |

---

## 训练配置

### 机器

| 项 | 规格 |
|---|---|
| 平台 | ModelScope Code Workspace（DSW-AMD） |
| 加速卡 | **AMD Instinct MI300A**（APU 统一内存） |
| 显存 | 191.7 GB |
| ROCm | 7.2.53211 |
| GPU 利用率 | 96–100%（全程满负载） |

### 耗时

| 阶段 | 时长 | tokens |
|---|---|---|
| Stage 1（基础训练） | 约 2.7 小时 | 2.0B |
| Stage 2（续训，补口语） | **5.47 小时** | 2.2B（累计 4.2B） |
| **纯训练合计** | **约 8.2 小时** | **4.2B** |

Stage 2 实测吞吐 **213K tok/s**，每步 **4.7 秒**（全程零抖动），
最终 loss **0.7393**。

### 超参

| 项 | Stage 1 | Stage 2 |
|---|---|---|
| LR 峰值 → 下限 | 1e-3 → 1e-4 | 3e-4 → 3e-5 |
| warmup | 100 | 60 |
| micro batch | 16 | 8 |
| grad accum | 8 | 16 |
| global batch | 524,288 tok/step | 524,288 tok/step |
| optimizer | AdamW (0.9, 0.95), wd=0.1, fused | 同左 |
| 精度 | bf16 autocast | 同左 |
| 步数 | 3,815 | 8,010 |

---

## 数据

| 项 | 值 |
|---|---|
| 原始语料 | 15.9 GB |
| 训练 tokens | 4.2B |
| 词表 | 32,000（自训） |

**Stage 1**：ModelScope `iic/WMT-zh-en`（6.36GB，~25M 句对）

**Stage 2**：14 个 OPUS 语料（8.0GB）+ 语用等价对（39.9M tokens）

→ 完整清单见 [`docs/数据集清单_OPUS.md`](docs/数据集清单_OPUS.md)

口语（OpenSubtitles/TED）与新闻（News-Commentary）**加权 ×3**。

**样本格式**：
```
句级：<|zh|> 中文 <|en|> English <|eos|>
段落：<|zh|> [8句拼接] <|en|> [对应英文段] <|eos|>
```

段落级样本用于训练 4–8K 长上下文——**机翻逐句翻，我们整段翻。**

---

## 评估结果

### 中译英 · 47 句短句 vs OPUS-MT

| 类别 | 句数 | ym 胜 | 机翻胜 | 平手 | 都错 |
|---|---|---|---|---|---|
| A 句末语气词 | 6 | 0 | 3 | 3 | 0 |
| B 体标记时态 | 5 | 0 | 3 | 2 | 0 |
| C 口语应答语 | 12 | 0 | 10 | 0 | 2 |
| D 感叹词 | 5 | 0 | 1 | 4 | 0 |
| E 话语标记 | 10 | 0 | 8 | 1 | 1 |
| F 委婉敬语 | 6 | 0 | 4 | 1 | 1 |
| **G 对照组** | 3 | 0 | 0 | **3** | 0 |
| **合计** | **47** | **0** | **29 (62%)** | **14 (30%)** | **4 (9%)** |

> **结论：OPUS-MT 完胜。** 但⚠️ **对照组 3:3 完全打平** ——
> 日常句子已经追平机翻，差距只在口语/语用。

### 长文本 · 13 例

ym 0 胜 / 机翻 8 胜 / 双方都错 4。

典型失败（诚实记录）：
- 段落-记叙：**死循环**，`he was a new engineer` 重复 68 次
- 一致性-术语：**只翻末半句**，前 60 字全漏
- 长句-技术：**双方都错**（我们把 4096 译成 four-thousand-ninety-six，机翻译成 `400096 medals`）

→ 完整逐句对比见 [`eval_results/ym_vs_机翻_完整对比.txt`](eval_results/ym_vs_机翻_完整对比.txt)

### 速度

| 场景 | ym-Translation | OPUS-MT (beam=4) |
|---|---|---|
| 短句（手机端最常用） | **0.19–0.29 s/句** | 0.20–0.99 s/句 |

---

## 已知问题

| 问题 | 状态 |
|---|---|
| 口语应答语（C 类 0:10） | ❌ 最大短板 |
| 英译中孤立语气词崩溃 | ❌ 语料泄漏 + 循环 |
| 长文本死循环 / 截断 | ❌ 需 beam search |

### 路线图

| 优先级 | 措施 |
|---|---|
| **P0** | 加 beam search + repetition penalty（治循环） |
| **P1** | 补口语语料（OpenSubtitles） |
| **P1** | 语用对 REPEAT 提到 5000+（**可反超机翻**） |
| **P2** | 长度惩罚 + 超长分段 |
| **P3** | 量化 int8/int4 + ONNX 导出 |

---

## 快速开始

```bash
pip install -r requirements.txt

# ── Stage 1 ──
python3 scripts/train_vocab.py                        # 训词表
python3 scripts/tokenize_trans.py                     # 分词
TARGET_TOKENS=2000000000 python3 scripts/train_trans.py

# ── Stage 2 ──
bash scripts/dl_new_corpora.sh                        # 下载 OPUS 语料
python3 scripts/build_pragmatic_pairs.py              # 构造语用等价对
python3 scripts/tokenize_new.py
bash scripts/start_stage2.sh

# ── 评估 ──
python3 scripts/eval_v3.py    --ckpt output/ckpt2/latest.pt
python3 scripts/eval_long.py  --ckpt output/ckpt2/latest.pt
```

---

## 踩坑记录（值得一读）

1. **LR 调度必须基于相对步数** —— 否则续训时 LR 立刻衰减到谷底
2. **`TARGET_TOKENS` 是累计总量**，不是新增量
3. **数据通配符 `part_*.bin` 匹配不到 `ord_*.bin`** —— 差点让 2.2B tokens 白跑
4. **吞吐口径陷阱** —— 日志 `213K tok/s` 的分子含历史存量，真实速度看步进速率
5. **中文必须去空格** —— 否则 BPE 把多字词切碎，token 效率掉 2.4 倍

详见 [`docs/环境备忘.md`](docs/环境备忘.md) 与 [`docs/Stage2训练完成报告.md`](docs/Stage2训练完成报告.md)。

---

## License

Apache-2.0

## 致谢

- 语料：[OPUS](https://object.pouta.csc.fi)、ModelScope `iic/WMT-zh-en`
- 基线：[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- 平台：ModelScope Code Workspace / AMD Instinct MI300A
