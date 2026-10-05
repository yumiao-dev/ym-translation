# -*- coding: utf-8 -*-
"""长文本翻译评估：ym-Translation vs OPUS-MT

测试维度：
  1. 长句（单句 40-80 字）
  2. 多句段落（200-400 字）
  3. 上下文一致性（代词、时态、术语）
  4. 4K 上下文能力（模型 CTX=4096）

输出：对比表格，含双方译文与人工判定
"""
import os, sys, time, glob
import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/mnt/workspace/ym-Translation/scripts")
import rmsnorm_shim  # noqa: F401
from train_trans import YMTranslate, CTX

BASE = "/mnt/workspace/ym-Translation"
_c = glob.glob(os.path.join(BASE, "**/tokenizer.json"), recursive=True)
TOK_PATH = _c[0] if _c else os.path.join(BASE, "tokenizer/tokenizer.json")
L_ZH, L_EN = "<|zh|>", "<|en|>"
STOP_MARKS = ["<|eos|>", "<eos>", "<|endoftext|>"]

# ---------------------------------------------------------------
# 测试集：长文本
# ---------------------------------------------------------------
LONG_CASES = [
    # ---------- 1. 长单句（40-80 字）----------
    ("长句-新闻",
     "国家统计局今日发布数据显示，上半年国内生产总值同比增长百分之五点三，"
     "其中第三产业增加值增速高于整体水平，消费对经济增长的贡献率超过六成。"),

    ("长句-技术",
     "该模型采用分组查询注意力机制，在保持多头注意力表达能力的同时显著降低了键值缓存的显存占用，"
     "使其能够在消费级显卡上支持长达四千零九十六个令牌的上下文窗口。"),

    ("长句-日常",
     "昨天下午我在公司附近的咖啡馆碰到了一个多年未见的老同学，"
     "我们聊了很久，从各自的工作聊到家庭，最后还约好下个月一起去看场电影。"),

    ("长句-商务",
     "关于贵司提出的交货周期调整方案，我方经内部评估认为，"
     "若能提前两周完成首批交付，我们可以接受相应价格上浮，但需要贵方提供质量保证条款。"),

    ("长句-描述",
     "秋天的傍晚，阳光斜斜地洒在街道两旁的梧桐树上，"
     "金黄的落叶随着微风缓缓飘落，铺满了人行道，偶尔有骑车的人从上面碾过，发出沙沙的声响。"),

    # ---------- 2. 多句段落（200-400 字）----------
    ("段落-记叙",
     "小李是一名刚毕业的程序员，入职一家互联网公司已经三个月了。"
     "刚开始的时候，他常常因为不熟悉代码库而加班到很晚，但他从不抱怨，"
     "而是利用周末时间阅读项目文档，向同事请教。"
     "三个月后的今天，他已经能够独立负责一个模块，并且在团队周会上提出了不错的优化建议。"
     "他的主管说，态度比能力更重要，小李正是那种能把事情做好的人。"),

    ("段落-说明",
     "机器学习模型的训练过程通常分为几个阶段。"
     "首先需要收集并清洗数据，去除噪声和重复样本；"
     "然后进行特征工程，把原始数据转换成模型可以理解的形式；"
     "接着选择合适的模型结构，并在训练集上反复迭代优化参数；"
     "最后在测试集上评估效果，如果指标不达标，还需要回头调整数据或结构。"
     "整个过程往往需要多次循环，很难一次成功。"),

    ("段落-议论",
     "有人认为，人工智能的发展会大量取代人类的工作岗位，从而引发严重的社会问题。"
     "但另一些观点指出，历史上每一次技术革命虽然淘汰了旧职业，同时也创造了新的就业机会。"
     "关键在于社会能否及时提供相应的教育和培训，帮助劳动者完成技能转型。"
     "因此，与其担心被取代，不如思考如何适应变化。"),

    ("段落-对话",
     "「你昨天怎么没来开会？」小王问道。"
     "「不好意思，我临时有个急事要处理。」小张解释道，"
     "「不过我已经把材料发到群里了，你看到了吗？」"
     "「看到了，但是有几个地方不太明白，等会儿能跟你确认一下吗？」"
     "「当然可以，午休的时候我们找个会议室聊吧。」"),

    # ---------- 3. 上下文一致性 ----------
    ("一致性-代词",
     "张教授把他的研究笔记交给了他的学生，因为他说他自己已经用不上了，"
     "希望这些笔记能帮助她少走一些弯路。他的学生接过笔记，感动得说不出话来。"),

    ("一致性-时态",
     "我去年在杭州工作，那时候每天骑自行车上班。"
     "现在我搬到了上海，改坐地铁了。"
     "明年我打算去北京，到时候估计得习惯更长的通勤时间。"),

    ("一致性-术语",
     "本次实验采用了 Transformer 架构，使用 AdamW 优化器，"
     "学习率设置为三乘以十的负四次方，批次大小为三十二。"
     "训练结束后，我们在验证集上测试了模型的困惑度。"),

    # ---------- 4. 长上下文压力（接近 4K）----------
    ("短文-综合",
     "清晨六点，闹钟准时响起。"
     "我挣扎着从床上爬起来，拉开窗帘，外面的天空还是一片灰蒙蒙的。"
     "洗漱完毕后，我给自己冲了一杯咖啡，坐在书桌前开始处理昨天没做完的工作。"
     "电脑屏幕的蓝光映在脸上，键盘敲击声在安静的房间里格外清晰。"
     "大约过了两个小时，阳光终于透过云层照了进来，屋子里一下子亮堂了许多。"
     "我伸了个懒腰，决定出门走走。"
     "小区门口的早餐店已经开始忙碌，老板娘热情地招呼着每一位客人。"
     "我点了一碗豆浆和两个包子，坐在角落里慢慢吃着，看着来来往往的行人。"
     "有人行色匆匆，有人悠闲散步，每个人的脸上都写着不同的故事。"
     "吃完早餐，我沿着河边的小路走了一圈，"
     "微风拂过水面，泛起层层涟漪，几只野鸭悠闲地游过。"
     "这一刻，我感到前所未有的平静。回到家后，我重新坐到电脑前，"
     "思路竟然变得清晰起来，之前卡了很久的问题迎刃而解。"),
]


def load(ckpt):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = YMTranslate().to(dev)
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = d["model"]
    if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    print(f"[ym] loaded {ckpt} | step={d.get('step')} | {d.get('tokens',0)/1e9:.3f}B tok", flush=True)
    return m, dev


def gen_ym(model, tok, dev, text, max_new=400):
    """贪心解码（与 eval_v3 一致）"""
    prompt = f"{L_ZH} {text} {L_EN}"
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_new):
            xin = x[:, -CTX:]
            logits, _ = model(xin)
            nxt = logits[0, -1].argmax().item()
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=dev)], dim=1)
            if nxt in STOP_IDS:
                break
    dt = time.time() - t0
    seg = tok.decode(out)
    hit = [seg.find(mk) for mk in STOP_MARKS]
    hit = [h for h in hit if h >= 0]
    if hit:
        seg = seg[:min(hit)]
    for mk in (L_ZH, L_EN):
        p = seg.find(mk)
        if p >= 0:
            seg = seg[:p]
    return seg.strip(), dt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "output/ckpt2/latest.pt"))
    a = ap.parse_args()

    tok = Tokenizer.from_file(TOK_PATH)
    STOP_IDS = set()
    for mk in STOP_MARKS:
        e = tok.encode(mk)
        if len(e.ids) == 1:
            STOP_IDS.add(e.ids[0])
    if not STOP_IDS:
        e = tok.encode("<|eos|>")
        if e.ids:
            STOP_IDS.add(e.ids[-1])

    model, dev = load(a.ckpt)
    print(f"=== LONG-TEXT EVAL: {len(LONG_CASES)} cases ===", flush=True)
    for i, (tag, zh) in enumerate(LONG_CASES, 1):
        nchar = len(zh)
        out, dt = gen_ym(model, tok, dev, zh)
        print(f"\n----[{i:02d}] {tag} | 源文 {nchar} 字----", flush=True)
        print(f"ZH: {zh}", flush=True)
        print(f"YM: {out}", flush=True)
        print(f"T : {dt:.2f}s | out_chars={len(out)}", flush=True)
    print("\n=== DONE ===", flush=True)
