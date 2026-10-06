# 评估数据说明

本目录包含 **FLORES-200 devtest 中译英（zho_Hans → eng_Latn，1012 句）** 的完整评测原始数据，
所有分数均由 `sacrebleu 2.6.0` 在 **AMD Instinct MI300X** 上计算。

## 文件

| 文件 | 说明 |
|---|---|
| `zh.txt` | 中文源句（1012 行） |
| `en.txt` | 英文参考译文（官方 gold，1012 行） |
| `ym_hyp.txt` | **ym-Translation Stage 3 (8.3B tok)** 输出 |
| `YM2_hyp.txt` | **ym-Translation Stage 2 (4.2B tok)** 输出（对照组） |
| `opus_hyp.txt` | **OPUS-MT (Helsinki-NLP/opus-mt-zh-en, 77.9M)** 输出（基线） |
| `score_*.json` | 三份 sacrebleu 评分结果 |

## 结果

| 模型 | BLEU (13a) | BLEU (zh) | chrF | chrF++ | TER | 输出词数 | 空输出 |
|---|---|---|---|---|---|---|---|
| ym-Translation **Stage 3** (8.3B) | **17.50** | 17.44 | **46.46** | **43.98** | 76.05 | 21.49 | 0 |
| ym-Translation Stage 2 (4.2B) | 11.80 | 9.76 | 41.77 | 39.27 | 103.30 | 27.35 | 0 |
| **OPUS-MT** (77.9M) | **23.16** | 23.07 | **53.60** | **50.97** | 65.74 | 21.79 | 0 |

参考译文平均 **21.64 词**。

## 复现方式

本目录**自带全部假设输出**，所以**分数可以直接复算**，无需 GPU：

```bash
pip install sacrebleu==2.6.0
python3 - <<'PY'
import sacrebleu
ref = open('en.txt', encoding='utf-8').read().splitlines()
for tag, path in [('YM_S3', 'ym_hyp.txt'),
                  ('YM_S2', 'YM2_hyp.txt'),
                  ('OPUS',  'opus_hyp.txt')]:
    hyp = open(path, encoding='utf-8').read().splitlines()
    print(tag,
          round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
          round(sacrebleu.corpus_chrf(hyp, [ref]).score, 2),
          round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
          round(sacrebleu.corpus_ter(hyp, [ref]).score, 2))
PY
```

> chrF++ 即 `corpus_chrf(word_order=2)`。

假设输出本身的**生成**过程需要训练 checkpoint 与 OPUS-MT 权重，**不在本目录内**。

## 生成配置

| 项 | ym-Translation | OPUS-MT |
|---|---|---|
| 解码 | 贪心（argmax） | beam search (beam=4) |
| max tokens | 192 | 256 |
| 精度 | bfloat16 | fp32 |
| batch | 1 | 32 |
| 速度 | **184 ms/句** | 34 ms/句 |

> **注意公平性**：ym 使用**贪心解码**，OPUS-MT 使用 **beam=4**（性能更高但更慢）。
> ym 尚未实现 beam search / repetition penalty，这是后续 P0 优化项。

## 数据来源

FLORES-200 官方包（Meta AI）：

```
https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
```

使用 `devtest/zho_Hans.devtest` 与 `devtest/eng_Latn.devtest`。
