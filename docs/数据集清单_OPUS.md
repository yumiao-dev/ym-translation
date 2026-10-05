# 中英平行语料 —— 完整可下载清单

> 探测时间 2026-10-05 | 可用源：OPUS官方(object.pouta.csc.fi) / hf-mirror.com

## 一、OPUS 官方源（直连，支持断点续传）★推荐
| 语料 | en-zh 大小 | 领域 | 质量 |
|---|---|---|---|
| **CCMatrix/v1** | **5875 MB** | 网页爬取，海量 | 中（需清洗） |
| **UNPC/v1.0** | **1536 MB** | 联合国文件，人工译 | 极高 |
| **ParaCrawl/v9** | **1307 MB** | 网页爬取 | 中 |
| **MultiUN/v1** | **900 MB** | 联合国，多领域 | 高 |
| **WikiMatrix/v1** | **104 MB** | 维基百科 | 高 |
| News-Commentary/v16 | 30 MB | 新闻评论 | 极高 |
| Tanzil/v1 | 17 MB | 古兰经 | 高 |
| TED2020/v1 | 1.1 MB | TED演讲 | 极高 |
| QED/v2.0a | 0.8 MB | 教育字幕 | 高 |
| PHP/v1 | 1.0 MB | 技术文档 | 高 |

**小计：约 9.9 GB**（去掉CCMatrix约4GB）

## 二、HuggingFace 镜像源（hf-mirror.com）
| 数据集 | 大小 | 说明 |
|---|---|---|
| **Mxode/BiST** | **29.2 GB** | 5962万句对，LLM生成+验证，自带dev |
| Helsinki-NLP/un_pc/en-zh | 2.55 GB | = UNPC |
| Helsinki-NLP/opus-100/en-zh | 0.14 GB | 100万句对，多源混合 |

## 三、已有
| iic/WMT-zh-en | 6.36 GB | 630万句对，已tokenize |

## 合计可获取：约 45-48 GB 原始语料

## 下载优先级建议
1. **BiST** (29.2G) —— 质量最高、量最大、自带验证集
2. **OPUS: UNPC + ParaCrawl + MultiUN + WikiMatrix** (3.8G) —— 官方高质量
3. **CCMatrix** (5.9G) —— 量最大但需激进清洗
4. 已有 WMT (6.36G)

→ 全下 = 约 45G → tokenize 后预计 **12-15B tokens**
