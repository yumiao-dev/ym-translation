# 原始日志归档

| 文件 | 说明 | 来源 |
|---|---|---|
| `ym_eval_long.log` | ym-Translation 长文本评估（13 例）原始输出 | 实例 `/tmp/elong.log` |
| `opus_eval_long.log` | OPUS-MT 长文本评估（13 例）原始输出 | 实例 `/tmp/olong.log` |

## Stage 2 训练日志（关键片段）

原始文件：实例 `/tmp/train2.log`（训练结束后实例已关闭，此处保留关键时间线）

```
[10-05 07:15:28] step 3818/8010 | loss 0.8454 | lr 1.50e-05 | 2.002B tok | 81564K tok/s | 0.4min
[10-05 07:15:52] step 3823/8010 | loss 0.9168 | lr 4.00e-05 | 2.004B tok | 41838K tok/s | 0.8min
[10-05 07:16:15] step 3828/8010 | loss 0.7511 | lr 6.50e-05 | 2.007B tok | 28156K tok/s | 1.2min
[10-05 07:16:38] step 3833/8010 | loss 0.8476 | lr 9.00e-05 | 2.010B tok | 21246K tok/s | 1.6min
[10-05 07:17:02] step 3838/8010 | loss 0.9748 | lr 1.15e-04 | 2.012B tok | 17067K tok/s | 2.0min
[10-05 07:17:25] step 3843/8010 | loss 0.8447 | lr 1.40e-04 | 2.015B tok | 14272K tok/s | 2.4min
[10-05 07:17:48] step 3848/8010 | loss 0.8587 | lr 1.65e-04 | 2.017B tok | 12266K tok/s | 2.7min
   ...（中间省略，训练全程每约 20 秒记一行，无抖动）...
[10-05 12:42:47] group0 tokens: 30 parts
[10-05 12:42:47] group1 tokens2: 64 parts
[10-05 12:42:51] step 8001/8010 | loss 0.7204 | lr 3.00e-05 | 4.195B tok | 213K tok/s | 327.8min
[10-05 12:43:15] step 8006/8010 | loss 0.7393 | lr 3.00e-05 | 4.197B tok | 213K tok/s | 328.2min
[10-05 12:43:36] ckpt saved @ step 8010 (4.200B tok) [keep 4]
[10-05 12:43:36] === STAGE2 DONE: step 8010, 4.200B tokens ===
```

**关键数据**：
- 总耗时 **328.2 分钟 = 5.47 小时**
- 步数 3818 → 8010
- 数据 `group0 tokens: 30 parts`（Stage1 语料）+ `group1 tokens2: 64 parts`（Stage2 新语料）
- 最终 loss **0.7393**，最终 lr **3.00e-05**

> 注：`K tok/s` 一列的数值在训练初期虚高（分子含历史 token 存量），
> 稳态值约 **111K tok/s**（4.7 秒/步 × 524,288 tok/步）。
