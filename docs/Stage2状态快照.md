# ym-Translation Stage 2 续训 · 状态记录

**更新时间**：2026-10-05 07:19
**实例**：`dsw-2230987-5c9f4b45fd-s7sf5`（DSW-AMD / AMD Instinct MI300A）
**规格**：ROCm 7.2.53211 · torch 2.12.0+git6bbd260 · HBM 191.7 GB

---

## 一、训练状态：正常运行中

```
tmux 会话：t2
日志：/tmp/train2.log
检查点：/mnt/workspace/ym-Translation/output/ckpt2/   （每 250 步存一次，保留 4 个）
```

启动日志（已确认正确）：

```
RESUMED from /mnt/workspace/ym-Translation/output/ckpt/latest.pt @ step 3814 (2.000B tok)
torch.compile OK
group0 tokens: 30 parts          ← 原 Stage1 数据
group1 tokens2: 64 parts         ← 新增口语/语气词数据（修复后）
START_STEP=3814 | 余下 4196 步 (2.20B tok)
=== stage2 training starts ===
step 3818/8010 | loss 0.8454 | lr 1.50e-05
```

性能实测：

| 指标 | 数值 |
|---|---|
| 每步耗时 | 4.7 秒（稳定） |
| 每步 token | 8 × 4096 × 16 = 524,288 |
| 有效吞吐 | ≈ 112K tok/s |
| GPU 利用率 | 96% |
| 显存占用 | 6%（远未吃满） |
| **剩余 4196 步预计耗时** | **≈ 5.5 小时** |
| 实例剩余额度 | ≈ 7.8 小时 |

结论：**余量充足，无需调整任何超参。**

---

## 二、本轮修复的关键 Bug

### Bug：新数据未加载（`group1 tokens2: 0 parts`）

**现象**：首次启动时日志显示 `group1 tokens2: 0 parts`，训练只吃老数据。

**根因**：`train_trans2.py` 的 `DATA_GLOBS` 写的是

```python
os.path.join(BASE_DIR, "data/tokens2/part_*.bin"),   # ← 通配符错
```

但 `data/tokens2/` 下实际文件名是：

- `ord_*.bin` —— 44 个（新语料分词产物）
- `prag_*.bin` —— 20 个（语用等价对产物）

`part_*.bin` 一个都匹配不到，该目录 8.4 GB / 64 个文件全部被忽略。

**修复**：改为 `data/tokens2/*.bin`

**修复后**：`group1 tokens2: 64 parts` ✅，吞吐也从 42K tok/s 升到 112K tok/s（文件数从 30 → 94，I/O 争抢大幅缓解）。

---

## 三、数据集构成

### 原有（`data/tokens/`，Stage 1，约 2.0B tokens）
`part_*.bin` × 30

### 新增（`data/tokens2/`，8.4 GB，约 2.4B tokens，64 个分片）

| 文件 | 数量 | 内容 |
|---|---|---|
| `ord_*.bin` | 44 | 14 个 OPUS 语料分词产物 |
| `prag_*.bin` | 20 | 语用等价对（定向构造） |

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

> 加权思路：口语类（OpenSubtitles / TED）和新闻类（News-Commentary）权重 ×3，
> 直指 C 类应答语 0/12 和 E 类话语标记 2/10 的短板。

**语用等价对**（`build_pragmatic_pairs.py`，核心创新）：

| 类别 | 条目数 |
|---|---|
| resp（应答语） | 556 |
| mark（话语标记） | 113 |
| modal（情态） | 32 |
| polite（委婉敬语） | 30 |
| interj（感叹词） | 30 |
| tense（体标记） | 21 |
| idiom（习语） | 19 |
| **合计（含反向）** | **1602** |

样本示例：

| 中文 | 英文 |
|---|---|
| 我知道了。 | I see. |
| 哪里哪里。 | You flatter me. |
| 随你便。 | Suit yourself. |
| 辛苦了。 | Thanks for your hard work. |
| 劳驾，借过一下。 | Excuse me, could I get through? |

生成方式：`REPEAT=1500` → 20 个 `prag_*.bin`，约 **39.9M tokens**。

> 设计理由：中文口语应答语是**语用固定说法**，不是字面组合。
> "我知道了" 字面是 "I know"，但实际语义是 "I see" —— 网页语料里几乎不存在这种对照，
> 必须定向构造 + 过采样。

---

## 四、超参配置

```bash
TARGET_TOKENS=4200000000   # 累计总量（含已训 2.0B）→ 总步数 8010
LR_PEAK=3e-4               # 续训降 3 倍（Stage1 是 1e-3）
LR_MIN=3e-5
WARMUP=60                  # 相对步数重新爬坡
```

模型：54.4M 参数 · 10 层 · hidden 512 · 8 头（KV 2 头）· FFN 2048 · vocab 32000 · ctx 4096
优化器：AdamW(fused) · betas=(0.9,0.95) · wd=0.1 · grad_clip=1.0
批次：micro_bs=8 × ctx=4096 × grad_accum=16

### LR 调度修复（重要）

续训时**必须基于相对步数**重新爬坡，否则：

```python
# 错误做法：用绝对 step
p = step / TOTAL_STEPS = 3814 / 8010 ≈ 0.47
# → LR 已在余弦中间，立刻衰减到谷底

# 正确做法：用相对步数
def get_lr(step, start_step):
    rel = step - start_step        # ← 关键
    if rel < WARMUP:
        return LR_PEAK * rel / WARMUP
    p = (rel - WARMUP) / max(1, (TOTAL_STEPS - start_step) - WARMUP)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + math.cos(min(p, 1.0) * math.pi))
```

---

## 五、当前效果基线（Stage 1 结果，待 Stage 2 后复测）

### 语气词专项测试（47 句 / 7 类）

| 结果 | 数量 | 占比 |
|---|---|---|
| 完全正确 | 15 | 32% |
| 将就可用 | 8 | 17% |
| 错误 | 24 | **51%** |

分类表现：

| 类别 | 说明 | 正确率 |
|---|---|---|
| G 对照组 | 普通陈述句 | **3/3 (100%)** |
| A 句末语气词 | 吗/呢/吧/啊 | 部分 |
| B 体标记 | 了/着/过 | 部分 |
| **C 应答语** | 我知道了/哪里哪里 | **0/12 (0%)** ❌ |
| D 感叹词 | 哎呀/哎哟 | 部分 |
| E 话语标记 | 说白了/话说回来 | 2/10 |
| F 委婉敬语 | 劳驾/麻烦您 | 部分 |

**两次循环失控**：

- `嗯嗯。` → `uh,uh,uh…`（贪心解码陷入重复）
- `要不……你还是别去了？` → `no,no,no…`

### 与 Helsinki-NLP opus-mt-zh-en 对比（同一批 47 句）

| | ym（54.4M） | OPUS-MT（144.5M） | 平手 | 都错 |
|---|---|---|---|---|
| 合计 | 0 | **29 (62%)** | 14 | 4 |

- OPUS-MT 参数量是我们的 **2.7 倍**，且用 beam=4
- **对照组完全打平**（3 句一样好）→ 基础翻译能力不输
- 差距集中在口语/虚词
- 速度我们快约 2 倍（0.19–0.29 vs 0.20–0.99 s/句）

**双方都栽的 4 句**：

| 中文 | ym | OPUS-MT |
|---|---|---|
| 我知道了 | I know | I know |
| 哪里哪里 | where | Where? Where?（更糟） |
| 差不多得了 | almost | almost |
| 那就打扰了 | — | — |

> **`我知道了 → I see` 是明确的反超方向**：OPUS-MT 到现在还错，
> 而我们的 `prag_*.bin` 里有 1500 次重复的这条对照，Stage 2 后极可能改对。

---

## 六、后续步骤

1. **等待训练完成**（约 5.5 小时，预计 12:50 左右）
2. 用 `eval_v3.py` 重跑 47 句，与 Stage 1 结果逐句对比
3. **重点验证 C 类应答语是否从 0/12 改善**
4. 验证两次循环失控是否消失
5. 推理侧尝试加 beam search（预期还能再涨一截）
6. 量化 int8/int4 → ONNX/TFLite → 安卓端接入

---

## 七、常用命令

```bash
# 看进度
tail -20 /tmp/train2.log

# 实时看
tail -f /tmp/train2.log

# 进 tmux 会话（退出：Ctrl+B 然后 D）
tmux attach -t t2

# 看 GPU
rocm-smi --showuse --showmemuse

# 看检查点
ls -lh /mnt/workspace/ym-Translation/output/ckpt2/
```

---

## 八、环境备忘

- **入口**：`https://www.modelscope.cn/code/workspace`（只需 ModelScope 登录态，绕开阿里云 DSW 的 40 分钟 cookie 过期）
- **cookie 注入顺序**：先 `inject_cookies.sh` → 再 `open(url)` → 再 reload，且 URL 必须带 `www`
- **NAS 共享**：`/mnt/workspace` 挂在同一天翼云 NAS，CPU 实例与 AMD 实例**零拷贝共享数据与检查点**
- **实例启动**：需要滑块验证，由用户手动完成
- **`torch` 版本差异**：CPU 实例是 2.3.1（无 `nn.RMSNorm`，需 `rmsnorm_shim.py`）；AMD 实例是 2.12.0（原生支持）
- **`tokenizers` 是 Rust 实现**：实测 3551K tok/s，不是 bug
