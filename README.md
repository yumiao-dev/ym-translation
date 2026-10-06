# ym-Translation

**中文 | [English](README_en.md)**

> 从零预训练的 **54.4M 参数**中英双向翻译小模型，目标是在安卓端侧**离线**运行。
> **单卡 AMD Instinct MI300A，纯训练 20 小时 40 分，8.3B tokens。**

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

## 模型权重

权重托管在 **Hugging Face**，本仓库只放代码与文档、**不含权重文件**：

| | |
|---|---|
| 🤗 仓库 | [yumiao-dev/ym-translation](https://huggingface.co/yumiao-dev/ym-translation) |
| 文件 | `model.safetensors`（208 MB, FP32, **Stage 3**）+ config + tokenizer |
| 一条命令下载 | `huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights` |

> 国内网络可先设镜像：`export HF_ENDPOINT=https://hf-mirror.com`

---

## 这是什么

一个**从零预训练**（非蒸馏、非微调）的中英双向翻译模型。作者的目标是做一个
**不联网、不调 API、完全跑在手机 NPU 上**的翻译模型。

区别于主流做法（拿大模型蒸馏），本项目选择 **从零预训练 + 自训 32K 词表**，
以在极小参数预算下榨出最大有效容量。

**当前状态：Stage 3。** FLORES-200 devtest 中译英 BLEU **17.50** / chrF++ **43.98**，
达工业级基线 OPUS-MT 的 **chrF++ 86.3%**；日常短句已能打平，口语语用仍是短板。
**详见[评估结果](#评估结果)一节——我们如实记录失败。**

---

## 发布件

| 文件 | 说明 |
|---|---|
| [`modeling_ym.py`](modeling_ym.py) | 模型定义，可独立加载权重（只依赖 torch） |
| [`infer.py`](infer.py) | 开箱即用推理脚本（CPU / GPU） |
| [`config.json`](config.json) | 架构参数 + 训练元信息 |
| [`tokenizer/`](tokenizer) | 自训 32K 联合 BPE 词表 |
| [`eval/`](eval) | FLORES-200 devtest 原始数据 + 三份 sacrebleu 评分（**可离线复算**） |
| [`FLORES-200_评测报告.md`](FLORES-200_评测报告.md) | 完整评测报告 |
| [`scripts/`](scripts) | 训练 / 分词 / 数据构造脚本 |
| [`docs/`](docs) | 环境备忘、数据集清单、训练报告 |
| [`eval_results/`](eval_results) | 逐句人工对比原始记录 |

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

Decoder-only Transformer，54.4M 参数（tied embedding 只计一次）。

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

> 导出的 `state_dict` 有 102 个 key，`emb.weight` 与 `head.weight` **共享同一张量**，
> 因此只保留 `emb.weight`；实际参数量 **54,406,912**。

---

## 训练配置

### 机器

| 项 | 规格 |
|---|---|
| 平台 | ModelScope Code Workspace（DSW-AMD） |
| 加速卡 | **AMD Instinct MI300A**（APU 统一内存） |
| 显存 | 191.7 GB |
| GPU 利用率 | 96–100%（全程满负载） |

### 耗时

| 阶段 | 步数区间 | tokens | GPU 时间 |
|---|---|---|---|
| Stage 1（基础训练） | 0 → 3,815 | 2.00B | 4h59m |
| Stage 2（续训，补口语） | 3,815 → 8,010 | 2.20B | 5h29m |
| 续训 | 8,010 → 13,506 | 2.88B | 7h10m |
| **Stage 3** | 13,506 → 15,830 | 1.22B | 3h02m |
| **合计** | **0 → 15,830** | **8.30B** | **20h40m** |

全程 **4.699 秒/步**（global batch = 524,288 tokens/step），
由 checkpoint 写入时间戳与训练日志**双向验证**，全程零抖动。
Stage 3 吞吐约 **760K tok/s**。

### 超参

| 项 | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| LR 峰值 → 下限 | 1e-3 → 1e-4 | 3e-4 → 3e-5 | 3e-4 → 3e-5 |
| warmup | 100 | 60 | 60 |
| micro batch × grad accum | 16 × 8 | 8 × 16 | 8 × 16 |
| global batch | 524,288 tok/step | 同左 | 同左 |
| optimizer | AdamW (0.9, 0.95), wd=0.1, fused | 同左 | 同左 |
| 精度 | bf16 autocast | 同左 | 同左 |

> 训练用 bf16 autocast，**导出权重为 FP32**（208 MB）。
> 端侧部署建议自行转 bf16/int8，体积可减半甚至更多。

---

## 数据

| 项 | 值 |
|---|---|
| 训练 tokens | 8.3B（15,830 步 × 524,288） |
| 词表 | 32,000（自训） |

| 阶段 | 步数区间 | tokens | 数据源 |
|---|---|---|---|
| Stage 1 | 0 → 3,815 | 2.00B | ModelScope `iic/WMT-zh-en`（6.36GB，~25M 句对） |
| Stage 2 | 3,815 → 8,010 | 2.20B | 14 个 OPUS 语料（8.0GB）+ **语用等价对** |
| 续训 | 8,010 → 13,506 | 2.88B | OPUS 扩展语料 |
| Stage 3 | 13,506 → 15,830 | 1.22B | OPUS 扩展 + 语用对增强 |

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

### FLORES-200 devtest · 中译英（1012 句）

| 指标 | **ym (Stage 3)** | ym (Stage 2) | OPUS-MT | ym/OPUS |
|---|---|---|---|---|
| **BLEU (13a)** | **17.50** | 11.80 | **23.16** | 75.6% |
| BLEU (zh) | 17.44 | 9.76 | 23.07 | 75.6% |
| **chrF** | **46.46** | 41.77 | **53.60** | 86.7% |
| **chrF++** | **43.98** | 39.27 | **50.97** | 86.3% |
| TER | 76.05 | 103.30 | 65.74 | — |
| 平均输出词数 | **21.49** | 27.35 | 21.79 | — |
| 空输出 | 0 | 0 | 0 | — |

参考译文平均 **21.64 词**。全部数据与评分在 [`eval/`](eval)，**可用 sacrebleu 离线复算**。

> **怎么读**：chrF++ 达基线 **86.3%**，说明**内容翻对了**；
> BLEU 只有 **75.6%**，说明**说得不够地道**。两者之差就是典型的小模型特征。
>
> ym 参数量只有 OPUS-MT 的 **70%**（54.4M vs 77.9M），且是**单模型双向**，
> 而 OPUS 每个方向各需一个模型；ym 还只用**贪心解码**对阵 beam=4——这对 ym 不利。

### Stage 2 → Stage 3：多花一倍的 tokens，换来了什么

| 指标 | Stage 2 (4.2B) | Stage 3 (8.3B) | 变化 |
|---|---|---|---|
| BLEU (13a) | 11.80 | **17.50** | **+48.3%** |
| chrF++ | 39.27 | **43.98** | +12.0% |
| TER | 103.30 | **76.05** | **−26.4%** |
| 平均输出词数 | 27.35 | **21.49** | −21.4% |
| 生成速度 | 1099 ms/句 | **184 ms/句** | **6.0×** |

Stage 2 输出 27.35 词，参考只有 21.64 词，TER 高达 103.3
（>100 意味着改动量比"照抄参考"还大）。Stage 3 把输出收敛到 21.49 词。

> **输出长度回归 + 速度提升 6 倍 = 同一个根因：复读被大幅抑制。**
> 这不是某项能力突然变强，而是长尾缺陷被系统性修复。

### 人工对比 · 47 句短句 vs OPUS-MT

| 指标 | Stage 2 | Stage 3 |
|---|---|---|
| ym 胜 | 0 | **2** |
| OPUS 胜 | 29 | **17** |
| 平手 | 12 | **27** |
| 都错 | 4 | **0** |

> 进步不是靠某一类突然变强，而是**全线把「都错」和「OPUS 独占胜」转成了「平」**。
> 逐句记录见 [`eval_results/`](eval_results)。

### 长文本

Stage 2 时期 13 例长文本：ym 0 胜 / 机翻 8 胜 / 双方都错 4，
典型失败包括**死循环**（`he was a new engineer` 重复 68 次）与**只翻后半句**。
Stage 3 复读已明显改善，但长文本仍是弱项。

### 生成配置（公平性说明）

| 项 | ym-Translation | OPUS-MT |
|---|---|---|
| 解码 | **贪心（argmax）** | beam search (beam=4) |
| max tokens | 192 | 256 |
| 精度 | bfloat16 | fp32 |
| batch | 1 | 32 |
| 速度 | **184 ms/句** | 34 ms/句 |

---

## 已知问题

| 问题 | 状态 |
|---|---|
| 词组重复（`medal medals` 类） | ⚠️ 已改善，仍存在 |
| 长文本复读 / 截断可能砍在句子中间 | ⚠️ 已改善 |
| 口语应答语、语用 | ❌ 最大短板 |
| 生僻词拼写（`exports` ← `exits`） | ❌ 未解 |

### 路线图

| 优先级 | 措施 |
|---|---|
| **P0** | 加 beam search（beam=4）+ repetition penalty |
| **P1** | 补口语语料（OpenSubtitles） |
| **P1** | 语用对 REPEAT 提到 5000+ |
| **P2** | 长度惩罚 + 超长分段 |
| **P3** | 量化 int8/int4 + ONNX 导出 |

---

## 快速开始

### 直接用 Stage 3 权重推理

```bash
pip install -r requirements.txt
huggingface-cli download yumiao-dev/ym-translation --local-dir ym-weights

python3 infer.py --model ym-weights "今天天气怎么样？"
# 中: 今天天气怎么样？
# 英: How's the weather today?

python3 infer.py --model ym-weights -i          # 交互模式
```

### 从零重训

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

### 复算评测分数（无需 GPU）

见 [`eval/README.md`](eval/README.md)——本仓自带全部假设输出，`sacrebleu` 直接复算。

---

## 踩坑记录（值得一读）

1. **LR 调度必须基于相对步数** —— 否则续训时 LR 立刻衰减到谷底
2. **`TARGET_TOKENS` 是累计总量**，不是新增量
3. **数据通配符 `part_*.bin` 匹配不到 `ord_*.bin`** —— 差点让 2.2B tokens 白跑
4. **吞吐口径陷阱** —— 日志里的 tok/s 分子含历史存量，真实速度看步进速率
5. **中文必须去空格** —— 否则 BPE 把多字词切碎，token 效率掉 2.4 倍

详见 [`docs/环境备忘.md`](docs/环境备忘.md) 与 [`docs/Stage2训练完成报告.md`](docs/Stage2训练完成报告.md)。

---

## 引用

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

## 致谢

- 语料：[OPUS](https://object.pouta.csc.fi)、ModelScope `iic/WMT-zh-en`
- 基线：[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- 平台：ModelScope Code Workspace / AMD Instinct MI300A