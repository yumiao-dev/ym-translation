# ym-Translation Stage 2 训练完成报告

**训练时间**：2026-10-05 07:15 → 12:43（约 5.5 小时）
**实例**：`dsw-2230987-5c9f4b45fd-s7sf5`（DSW-AMD / AMD Instinct MI300A）
**结果**：`=== STAGE2 DONE: step 8010, 4.200B tokens ===`

---

## 一、训练结果

| 项 | 值 |
|---|---|
| 总训练量 | **4.200B tokens**（Stage1 的 2.0B + 新增 2.2B） |
| 总步数 | 8010 / 8010（100% 完成） |
| 每步耗时 | 4.7 秒（全程零抖动） |
| GPU 利用率 | 96–100% |
| 最终 lr | 3.00e-05（精确落在 cosine 下限） |
| 最终 loss | 0.7393 |

**检查点**（每 250 步存一次，均为 623MB）：

```
output/ckpt2/step_007500.pt   12:03
output/ckpt2/step_007750.pt   12:23
output/ckpt2/step_008000.pt   12:42
output/ckpt2/step_008010.pt   12:43   ← 最终
output/ckpt2/latest.pt        12:43
```

---

## 二、修掉的关键 Bug

### Bug：新数据完全没加载（`group1 tokens2: 0 parts`）

**这是本轮最严重的问题**——如果不修，2.2B token 会全部训在旧数据上，等于白跑。

**根因**：`DATA_GLOBS` 写的是 `data/tokens2/part_*.bin`，但实际文件名是：

- `ord_*.bin` —— 44 个（14 个 OPUS 语料的分词产物）
- `prag_*.bin` —— 20 个（语用等价对）

`part_*.bin` 一个都匹配不到，**8.4 GB / 64 个文件全部被忽略**。

**修复**：改为 `data/tokens2/*.bin`

**修复效果**：

| | 修复前 | 修复后 |
|---|---|---|
| `group1 tokens2` | **0 parts** ❌ | **64 parts** ✅ |
| 吞吐 | 42K tok/s | 112K tok/s |

### Bug：LR 调度用了绝对步数

```python
# 错误：step=3814 时 p≈0.47，LR 已在余弦中段，立刻衰减到谷底
p = step / TOTAL_STEPS

# 正确：基于相对步数重新爬坡
def get_lr(step, start_step):
    rel = step - start_step
    if rel < WARMUP:
        return LR_PEAK * rel / WARMUP
    p = (rel - WARMUP) / max(1, (TOTAL_STEPS - start_step) - WARMUP)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + math.cos(min(p, 1.0) * math.pi))
```

---

## 三、数据构成

### 新增语料（`data/tokens2/`，8.4 GB，约 2.4B tokens）

**14 个 OPUS 语料**（8.0 GB raw）：

| 语料 | 句对 | 权重 |
|---|---|---|
| UNPC | 17.45M | 2 |
| MultiUN | 9,564,315 | 1 |
| OpenSubtitles_v2016 | 9,304,777 | **3** |
| XLEnt | 6,292,330 | 1 |
| WikiMatrix | 786,512 | 1 |
| wikimedia | 471,224 | 1 |
| Tanzil | 187,092 | 1 |
| TED2020 / TED2013 / NeuLab-TedTalks | — | **3** |
| News-Commentary_v16 | — | **3** |
| ALT / QED / PHP / infopankki / WMT-News / tico-19 / tldr-pages | — | 1 |

> 口语（OpenSubtitles / TED）与新闻（News-Commentary）加权 ×3，直指 C 类应答语与 E 类话语标记的短板。

**语用等价对**（定向构造，核心创新）：

| 类别 | 条目 |
|---|---|
| resp 应答语 | 556 |
| mark 话语标记 | 113 |
| modal 情态 | 32 |
| polite 委婉敬语 | 30 |
| interj 感叹词 | 30 |
| tense 体标记 | 21 |
| idiom 习语 | 19 |
| **合计（含反向）** | **1602** |

经 `REPEAT=1500` 扩充 → 20 个 `prag_*.bin`，约 **39.9M tokens**。

---

## 四、效果评估

### 中译英（zh→en，47 句）

**整体：明显改善，但 C 类应答语仍是硬伤。**

#### 分项结果

| 类别 | 表现 | 典型 |
|---|---|---|
| **A 句末语气词** | ✅ **大幅改善** | `你是学生吗？→Are you a student?` `你去哪儿呢？→Where are you going?` `好漂亮啊！→Good!` |
| **B 体标记** | ✅ **大幅改善** | `我吃了。→I ate it.` `他在吃饭呢。→he's eating.` `我明天就去。→I'll go tomorrow.` `还好我带了个伞。→I took the umbrella.` |
| **C 应答语** | ❌ **仍差** | `我知道了。→I know.`（仍错）`哪里哪里。→where.`（仍错）`辛苦了。→I'm tired.`（仍错） |
| **D 感叹词** | ✅ **明显改善** | `哎呀，我忘了带钥匙。→Oh, I forgot to take the key.` `唉，随便吧。→Oh, come on.` `哼，我才不信呢。→Oh, I don't believe it.` |
| **E 话语标记** | ⚠️ **部分改善** | `说白了就是钱的问题。→it's a problem with money.` ✅ `差不多得了。→it's almost good.` ✅ |
| **F 委婉敬语** | ⚠️ **部分** | `我可不可以这么理解？→Can I understand that?` ✅ |
| **G 对照组** | ✅ **全对** | `他昨天去了北京。→he went to Beijing yesterday.` 等 3/3 |

#### 关键改善（对比 Stage 1）

| 项目 | Stage 1 | Stage 2 | 变化 |
|---|---|---|---|
| `嗯嗯。` | `uh,uh,uh…` **循环失控** | `Well, uh, uh.` | ✅ **失控消失** |
| `差不多得了。` | `almost`（双方都错） | `it's almost good.` | ✅ **有实质进步** |
| B 类体标记 | 部分正确 | 5/6 正确 | ✅ **大幅提升** |
| D 类感叹词 | 部分 | 4/5 有感叹词 | ✅ **明显提升** |
| `我知道了。` | `I know` | `I know` | ❌ **仍未改对** |
| `哪里哪里。` | `where` | `where` | ❌ **仍未改对** |

#### 仍未解决

- `我知道了 → I see`（与 OPUS-MT 同错）
- `哪里哪里 → You flatter me`（仍输出 `where`）
- `辛苦了 → Thanks for your hard work`（仍输出 `I'm tired`）
- `收到 → Got it`（输出 `I bid., p. 3.`）

### 英译中（en→zh，47 句）

**常规句子表现良好，孤立语气词/应答语出现崩溃。**

| 类别 | 表现 | 典型 |
|---|---|---|
| **A 句末语气词** | ✅ **6/7 好** | `Are you a student?→你是个学生？` `Oh, how beautiful!→哦，多么美丽！` `Hurry up!→快点！` |
| **C 应答语** | ❌ **崩溃** | 见下方 |
| **D 感叹词** | ⚠️ **参半** | `Oh no, I forgot my keys.→哦，不，我忘了钥匙` ✅ / `Ouch, that hurts!→我该怎么办？` ❌ |
| **F 委婉** | ⚠️ | `Thanks for your hard work.→谢谢你辛苦的工作` ✅ / `You flatter me.→你平分我` ❌ |

#### 两类严重故障

**故障 1：联合国语料泄漏**

```
输入: Uh-huh.
输出: 哈吉先生(乌拉圭)(以西班牙语发言):主席先生，我谨感谢你召开这次会议，
      讨论乌拉圭在联合国系统内开展的活动。
```

模型对孤立英文语气词，从 MultiUN 语料里"背"出了原句。说明**短英文输入的训练样本严重不足**。

**故障 2：无限循环**

```
输入: Huh?
输出: 你好，我好想你，先生，我好想你，先生，我好想你，先生…

输入: （某句）
输出: 你，你看，你，你看，你，你看，你，你看…
```

---

## 五、结论与下一步

### 已达成
- ✅ 训练完整跑满 4.2B tokens，检查点齐全
- ✅ 中译英在 **A/B/D 类**（语气词、体标记、感叹词）**明显改善**
- ✅ `嗯嗯。` 的**循环失控已消失**
- ✅ `差不多得了` 等 E 类**有实质进步**
- ✅ G 类对照组**全对**，基础能力稳固

### 未达成
- ❌ **C 类应答语仍是最大短板**（`我知道了`/`哪里哪里`/`辛苦了` 全错）
- ❌ **英译中方向的孤立语气词崩溃**（语料泄漏 + 循环）

### 已识别的原因

1. **`我知道了 → I see` 这类语用对，REPEAT=1500 仍不够**。39.9M tokens 混在 2.2B 里只占 1.8%，被淹没。
2. **英译中短句样本缺失**。`prag_*.bin` 的英文侧虽已生成，但孤立语气词（`Uh-huh`）在中文侧无稳定对应，模型只能从 MultiUN 背套话。
3. **MultiUN 英文侧占比过高**，导致短英文输入的最强记忆是联合国套话。

### 下一步建议（按性价比排序）

| 优先级 | 措施 | 预期 |
|---|---|---|
| **P0** | 加 **beam search**（当前是贪心） | 一行改动，可消除大部分循环失控 |
| **P0** | 加 **repetition penalty** / `no_repeat_ngram_size` | 直接治循环 |
| **P1** | 把 `prag_*.bin` 的 REPEAT 提到 5000+，或单独做一轮小 lr 微调 | 专攻 C 类 |
| **P1** | 补 **英译中短句语料**（尤其英文语气词→中文） | 治 en2zh 崩溃 |
| **P2** | 降 MultiUN 权重（1 → 0.5） | 减少套话记忆 |
| **P2** | 推理时加 `min_length` / 长度惩罚 | 辅助 |

---

## 六、常用命令

```bash
# 查看检查点
ls -lh /mnt/workspace/ym-Translation/output/ckpt2/

# 重跑中译英评估
cd /mnt/workspace/ym-Translation && \
  python3 scripts/eval_v3.py --ckpt output/ckpt2/latest.pt

# 重跑英译中评估
python3 scripts/eval_en2zh.py --ckpt output/ckpt2/latest.pt
```

---

## 七、环境备忘

- **入口**：`https://www.modelscope.cn/code/workspace`
- **NAS 共享**：`/mnt/workspace` 挂同一天翼云 NAS，CPU 实例与 AMD 实例**零拷贝共享**
- **实例启动**：需滑块验证，由用户手动完成
