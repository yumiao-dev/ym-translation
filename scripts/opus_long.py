# -*- coding: utf-8 -*-
"""OPUS-MT 长文本对比：用同一批 LONG_CASES 跑 Helsinki-NLP opus-mt-zh-en"""
import os, sys, time, json
sys.path.insert(0, "/root/.codebuddy/artifact/2f8a7d30-9c21-4f0b-1d3c-2d6f2d4c1000")

import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = "/root/.codebuddy/artifact/2f8a7d30-9c21-4f0b-1d3c-2d6f2d4c1000"
MT_DIR = os.path.join(HERE, "opusmt")

# 复用 eval_long.py 里的测试集（把 LONG_CASES 内联，避免 import 触发 torch 模型加载）
LONG_CASES = [
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

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[opus] device={dev}", flush=True)
    tok = MarianTokenizer.from_pretrained(MT_DIR, local_files_only=True)
    mdl = MarianMTModel.from_pretrained(MT_DIR, local_files_only=True).to(dev).eval()
    nparam = sum(p.numel() for p in mdl.parameters())
    print(f"[opus] params={nparam/1e6:.1f}M", flush=True)

    print(f"=== OPUS LONG-TEXT EVAL: {len(LONG_CASES)} cases ===", flush=True)
    for i, (tag, zh) in enumerate(LONG_CASES, 1):
        t0 = time.time()
        enc = tok([zh], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            out = mdl.generate(**enc, num_beams=4, max_new_tokens=400,
                               no_repeat_ngram_size=3, early_stopping=True)
        dt = time.time() - t0
        txt = tok.decode(out[0], skip_special_tokens=True)
        print(f"\n----[{i:02d}] {tag} | 源文 {len(zh)} 字----", flush=True)
        print(f"ZH  : {zh}", flush=True)
        print(f"OPUS: {txt}", flush=True)
        print(f"T   : {dt:.2f}s | out_chars={len(txt)}", flush=True)
    print("\n=== OPUS DONE ===", flush=True)
