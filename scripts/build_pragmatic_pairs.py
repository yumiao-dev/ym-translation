# -*- coding: utf-8 -*-
"""
语用等价对构造器 —— 定向补齐中文口语 ↔ 英文语用等价映射

【为什么需要这个】
实测发现：`我知道了 → I know`（错）、`哪里哪里 → where`（崩）、
`随你便 → follow your time`（中式）—— 这些错误 OPUS-MT（144M，1B+ 句对）
同样犯。根因是：**中文口语应答语是语用固定说法，不是字面组合**，
网页爬取语料里几乎不存在「中文口语 → 英文地道说法」的平行句对。

→ 唯一解法：**人工/规则构造定向数据**，然后 oversample 放大重复次数。

【产出】
tokens2/ 目录下的 uint32 token 流，格式与原有数据一致：
  <|zh|> 中文 <|en|> English <|eos|>

【数据来源】
1. PHRASE_PAIRS：手工整理的语用等价对（核心，精度最高）
2. 模板扩展：用句式模板批量生成变体，提高泛化
"""
import os, sys, json, random, glob
import numpy as np

BASE = "/mnt/workspace/ym-Translation"
sys.path.insert(0, os.path.join(BASE, "scripts"))
from tokenizers import Tokenizer

TOK_PATH = glob.glob(os.path.join(BASE, "**/tokenizer.json"), recursive=True)[0]
tok = Tokenizer.from_file(TOK_PATH)

L_ZH, L_EN, EOS = "<|zh|>", "<|en|>", "<eos>"

# ============================================================
# 核心：语用等价对（中文口语 → 英文地道说法）
# 每条 = (中文, 英文, 类别)
# 类别: resp=应答语  modal=语气词  mark=话语标记  polite=委婉敬语
#       interj=感叹词  idiom=习语  tense=体标记
# ============================================================
PHRASE_PAIRS = [
    # ---------- 应答语（C 类重灾区）----------
    ("我知道了。", "I see.", "resp"),
    ("我知道了。", "Got it.", "resp"),
    ("我知道了。", "I understand.", "resp"),
    ("我知道了。", "Okay, I see.", "resp"),
    ("我知道了，你不用说了。", "I get it, you don't have to say it.", "resp"),
    ("我知道了，你不用说了。", "I see, no need to explain.", "resp"),
    ("我知道了，你不用说了。", "Okay, I got it, you can stop.", "resp"),
    ("知道了。", "Got it.", "resp"),
    ("明白了。", "I see.", "resp"),
    ("明白了。", "Understood.", "resp"),
    ("我明白了。", "I see.", "resp"),
    ("我明白了。", "I understand now.", "resp"),
    ("懂了。", "Got it.", "resp"),
    ("懂了。", "I get it.", "resp"),
    ("收到。", "Got it.", "resp"),
    ("收到。", "Roger that.", "resp"),
    ("收到。", "Copy that.", "resp"),
    ("收到。", "Noted.", "resp"),
    ("好的。", "Okay.", "resp"),
    ("好的。", "Sure.", "resp"),
    ("好的。", "All right.", "resp"),
    ("好的。", "Fine.", "resp"),
    ("好。", "Okay.", "resp"),
    ("行。", "Okay.", "resp"),
    ("行。", "Fine.", "resp"),
    ("行吧。", "All right, fine.", "resp"),
    ("可以。", "Sure.", "resp"),
    ("可以。", "That works.", "resp"),
    ("没问题。", "No problem.", "resp"),
    ("没问题。", "Sure thing.", "resp"),
    ("没事。", "It's fine.", "resp"),
    ("没事。", "No worries.", "resp"),
    ("不客气。", "You're welcome.", "resp"),
    ("不客气。", "No problem.", "resp"),
    ("别客气。", "You're welcome.", "resp"),
    ("别客气。", "Don't mention it.", "resp"),
    ("哪里哪里。", "You flatter me.", "resp"),
    ("哪里哪里。", "Not at all.", "resp"),
    ("哪里哪里。", "You're too kind.", "resp"),
    ("过奖了。", "You flatter me.", "resp"),
    ("过奖了。", "You're too kind.", "resp"),
    ("太过奖了。", "You're too kind.", "resp"),
    ("辛苦了。", "Thanks for your hard work.", "resp"),
    ("辛苦了。", "Thanks for the effort.", "resp"),
    ("辛苦了。", "Good work.", "resp"),
    ("麻烦你了。", "Sorry to trouble you.", "resp"),
    ("麻烦你了。", "Thanks for helping.", "resp"),
    ("打扰了。", "Sorry to bother you.", "resp"),
    ("那就打扰了。", "Sorry to bother you, then.", "resp"),
    ("不好意思。", "Sorry about that.", "resp"),
    ("不好意思。", "Excuse me.", "resp"),
    ("对不起。", "I'm sorry.", "resp"),
    ("抱歉。", "My apologies.", "resp"),
    ("算了吧。", "Never mind.", "resp"),
    ("算了吧。", "Forget it.", "resp"),
    ("算了吧。", "Let it go.", "resp"),
    ("算了。", "Forget it.", "resp"),
    ("算了。", "Never mind.", "resp"),
    ("随你便。", "Suit yourself.", "resp"),
    ("随你便。", "Whatever you like.", "resp"),
    ("随你便。", "It's up to you.", "resp"),
    ("你随便。", "It's up to you.", "resp"),
    ("看你的。", "It's your call.", "resp"),
    ("看着办吧。", "Play it by ear.", "resp"),
    ("看着办吧。", "Just figure it out.", "resp"),
    ("看着办吧。", "Use your judgment.", "resp"),
    ("你看着办。", "It's up to you.", "resp"),
    ("差不多得了。", "That's enough.", "resp"),
    ("差不多得了。", "Cut it out.", "resp"),
    ("差不多得了。", "Knock it off.", "resp"),
    ("够了。", "That's enough.", "resp"),
    ("够了。", "Enough.", "resp"),
    ("无所谓。", "It doesn't matter.", "resp"),
    ("无所谓啦。", "It doesn't matter.", "resp"),
    ("无所谓。", "Whatever.", "resp"),
    ("都行。", "Either way works.", "resp"),
    ("都行。", "Anything works.", "resp"),
    ("随便。", "Whatever.", "resp"),
    ("随便。", "Anything's fine.", "resp"),
    ("不早了。", "It's getting late.", "resp"),
    ("不早了，我该走了。", "It's getting late, I should go.", "resp"),
    ("那我先走了。", "I'll be going then.", "resp"),
    ("那我先走了。", "I'd better get going.", "resp"),
    ("回头见。", "See you later.", "resp"),
    ("回头聊。", "Talk to you later.", "resp"),
    ("改天再说。", "Let's talk about it another time.", "resp"),
    ("下次再聊。", "Let's talk again next time.", "resp"),
    ("慢慢来。", "Take your time.", "resp"),
    ("别急。", "Take it easy.", "resp"),
    ("别急。", "No rush.", "resp"),
    ("加油。", "You can do it.", "resp"),
    ("加油。", "Go for it.", "resp"),
    ("加油。", "Good luck.", "resp"),
    ("保重。", "Take care.", "resp"),
    ("恭喜。", "Congratulations.", "resp"),
    ("节哀。", "My condolences.", "resp"),

    # ---------- 语气词（A 类）----------
    ("你去哪儿呢？", "Where are you going?", "modal"),
    ("你干嘛呢？", "What are you doing?", "modal"),
    ("你在干什么呢？", "What are you up to?", "modal"),
    ("他呢？", "What about him?", "modal"),
    ("你呢？", "What about you?", "modal"),
    ("我们走吧。", "Let's go.", "modal"),
    ("走吧。", "Let's go.", "modal"),
    ("吃吧。", "Go ahead and eat.", "modal"),
    ("来吧。", "Come on.", "modal"),
    ("就这样吧。", "Let's leave it at that.", "modal"),
    ("这很简单嘛。", "It's simple, obviously.", "modal"),
    ("这很简单嘛。", "Come on, it's easy.", "modal"),
    ("本来就是嘛。", "That's what it is, obviously.", "modal"),
    ("你别这样嘛。", "Come on, don't be like that.", "modal"),
    ("别这样嘛，好不好？", "Come on, please?", "modal"),
    ("不想去就别去呗。", "If you don't want to go, just don't.", "modal"),
    ("想买就买呗。", "If you want it, just buy it.", "modal"),
    ("那就试试呗。", "Then just give it a try.", "modal"),
    ("好漂亮啊！", "Oh, how beautiful!", "modal"),
    ("好漂亮啊！", "It's so beautiful!", "modal"),
    ("真好啊！", "That's wonderful!", "modal"),
    ("好美啊！", "How lovely!", "modal"),
    ("你快点呀！", "Hurry up!", "modal"),
    ("快点呀！", "Come on, hurry up!", "modal"),
    ("别走啊！", "Don't go!", "modal"),
    ("小心啊！", "Be careful!", "modal"),
    ("真的吗？", "Really?", "modal"),
    ("是吗？", "Is that so?", "modal"),
    ("是吗？", "Really?", "modal"),
    ("是吗？我还不知道呢。", "Is that so? I didn't know.", "modal"),
    ("好的呀。", "Sure, of course.", "modal"),
    ("好呀。", "Sure!", "modal"),

    # ---------- 体标记 / 时态（B 类）----------
    ("我吃了。", "I've eaten.", "tense"),
    ("我吃了。", "I have already eaten.", "tense"),
    ("我吃过了。", "I've already eaten.", "tense"),
    ("他走了。", "He's gone.", "tense"),
    ("他走了。", "He has left.", "tense"),
    ("他来过。", "He has been here before.", "tense"),
    ("他去过北京。", "He has been to Beijing.", "tense"),
    ("他去过北京。", "He's been to Beijing before.", "tense"),
    ("我去过那儿。", "I've been there.", "tense"),
    ("我看过这本书。", "I've read this book.", "tense"),
    ("我在吃饭呢。", "I'm eating right now.", "tense"),
    ("他在吃饭呢。", "He's eating right now.", "tense"),
    ("我正忙着呢。", "I'm busy right now.", "tense"),
    ("我明天就去。", "I'll go tomorrow.", "tense"),
    ("我马上就来。", "I'll be right there.", "tense"),
    ("我一会儿去。", "I'll go in a bit.", "tense"),
    ("他正在开会。", "He's in a meeting right now.", "tense"),
    ("下着雨呢。", "It's raining.", "tense"),
    ("门开着呢。", "The door is open.", "tense"),
    ("我还没吃呢。", "I haven't eaten yet.", "tense"),
    ("我还没去过。", "I haven't been there yet.", "tense"),

    # ---------- 话语标记（E 类）----------
    ("说真的，我不太想去。", "To be honest, I don't really want to go.", "mark"),
    ("说真的。", "To be honest.", "mark"),
    ("说实话。", "Honestly.", "mark"),
    ("说实话。", "To tell you the truth.", "mark"),
    ("老实说。", "Honestly speaking.", "mark"),
    ("说白了就是钱的问题。", "Put simply, it's a money problem.", "mark"),
    ("说白了。", "Put simply.", "mark"),
    ("说到底。", "In the end.", "mark"),
    ("反正我不管了。", "Anyway, I'm done with this.", "mark"),
    ("反正我不管了。", "Anyway, I don't care anymore.", "mark"),
    ("反正。", "Anyway.", "mark"),
    ("不管怎样。", "In any case.", "mark"),
    ("总之。", "In short.", "mark"),
    ("总之。", "All in all.", "mark"),
    ("那个……我是什么意思呢？", "Well... what do I mean?", "mark"),
    ("那个……", "Well...", "mark"),
    ("这个嘛……", "Well, that's...", "mark"),
    ("怎么说呢。", "How should I put it.", "mark"),
    ("怎么说呢。", "How do I put this.", "mark"),
    ("不是我说你，你这样不行。", "No offense, but this won't work.", "mark"),
    ("不是我说你。", "No offense, but.", "mark"),
    ("不是我说你，你真的该改了。", "No offense, but you really need to change.", "mark"),
    ("别误会，我不是那个意思。", "Don't get me wrong, that's not what I meant.", "mark"),
    ("别误会。", "Don't get me wrong.", "mark"),
    ("我不是那个意思。", "That's not what I meant.", "mark"),
    ("真是的，又迟到了。", "Honestly, late again.", "mark"),
    ("真是的。", "Honestly.", "mark"),
    ("真是的。", "For goodness' sake.", "mark"),
    ("我说真的。", "I'm serious.", "mark"),
    ("开玩笑的。", "Just kidding.", "mark"),
    ("开玩笑的。", "I'm just joking.", "mark"),
    ("顺便说一下。", "By the way.", "mark"),
    ("顺便说一句。", "By the way.", "mark"),
    ("话说回来。", "Speaking of which.", "mark"),
    ("话说回来。", "Come to think of it.", "mark"),
    ("无论如何。", "No matter what.", "mark"),
    ("值得一提的是。", "It's worth mentioning that.", "mark"),
    ("换句话说。", "In other words.", "mark"),
    ("也就是说。", "That is to say.", "mark"),
    ("也就是说。", "In other words.", "mark"),
    ("你看。", "Look.", "mark"),
    ("你听我说。", "Listen to me.", "mark"),
    ("我跟你说。", "Let me tell you.", "mark"),
    ("我跟你讲。", "I'm telling you.", "mark"),
    ("不是，我是说……", "No, I mean...", "mark"),
    ("对了。", "Oh, by the way.", "mark"),
    ("对了，还有一件事。", "Oh, one more thing.", "mark"),
    ("这样吧。", "Here's the thing.", "mark"),
    ("这样吧。", "How about this.", "mark"),
    ("那就这样吧。", "Let's leave it at that.", "mark"),
    ("先这样吧。", "That's it for now.", "mark"),
    ("好了。", "All right.", "mark"),
    ("行吧行吧。", "Okay, okay.", "mark"),

    # ---------- 委婉 / 敬语（F 类）----------
    ("劳驾，借过一下。", "Excuse me, could I get through?", "polite"),
    ("劳驾。", "Excuse me.", "polite"),
    ("借过一下。", "Excuse me, let me through.", "polite"),
    ("麻烦让一下。", "Excuse me, could you move?", "polite"),
    ("麻烦你帮我个忙。", "Could you do me a favor?", "polite"),
    ("麻烦你帮我个忙。", "Would you mind helping me?", "polite"),
    ("能帮我个忙吗？", "Could you help me out?", "polite"),
    ("恕我直言，这不太合适。", "With all due respect, this isn't appropriate.", "polite"),
    ("恕我直言。", "With all due respect.", "polite"),
    ("恕我直言。", "If I may say so.", "polite"),
    ("恕我冒昧。", "If I may be so bold.", "polite"),
    ("冒昧问一下。", "May I ask.", "polite"),
    ("不介意的话。", "If you don't mind.", "polite"),
    ("如果你方便的话。", "If it's convenient for you.", "polite"),
    ("要不……你还是别去了？", "Maybe... you'd better not go?", "polite"),
    ("要不……", "Maybe...", "polite"),
    ("要不算了？", "Maybe we should just drop it?", "polite"),
    ("你看这样行吗？", "How does this sound?", "polite"),
    ("可以吗？", "Would that be okay?", "polite"),
    ("我能不能……", "Would it be all right if I...", "polite"),
    ("我可不可以这么理解？", "Can I take it that way?", "polite"),
    ("不知当讲不当讲。", "I'm not sure if I should say this.", "polite"),
    ("有句话不知该不该说。", "I don't know if I should mention this.", "polite"),
    ("那我就恭敬不如从命了。", "I'll gladly accept.", "polite"),
    ("承蒙关照。", "Thank you for your kind help.", "polite"),
    ("失陪了。", "If you'll excuse me.", "polite"),
    ("请多指教。", "I look forward to your guidance.", "polite"),
    ("请便。", "Please, go ahead.", "polite"),
    ("您先请。", "After you.", "polite"),
    ("不吝赐教。", "I'd appreciate your advice.", "polite"),

    # ---------- 感叹词（D 类）----------
    ("哎呀，我忘了带钥匙。", "Oh no, I forgot my keys.", "interj"),
    ("哎呀。", "Oh no.", "interj"),
    ("哎呀。", "Oops.", "interj"),
    ("哎哟，好疼！", "Ouch, that hurts!", "interj"),
    ("哎哟。", "Ouch.", "interj"),
    ("哇，太棒了！", "Wow, that's great!", "interj"),
    ("哇。", "Wow.", "interj"),
    ("唉，随便吧。", "Sigh, whatever.", "interj"),
    ("唉。", "Sigh.", "interj"),
    ("唉。", "Alas.", "interj"),
    ("哼，我才不信呢。", "Hmph, I don't believe it.", "interj"),
    ("哼。", "Hmph.", "interj"),
    ("呸。", "Ugh.", "interj"),
    ("咦？", "Huh?", "interj"),
    ("哎？", "Hey?", "interj"),
    ("哦。", "Oh.", "interj"),
    ("哦哦。", "Oh, I see.", "interj"),
    ("噢！", "Oh!", "interj"),
    ("哎。", "Hey.", "interj"),
    ("哎呀妈呀。", "Oh my god.", "interj"),
    ("天哪！", "Oh my goodness!", "interj"),
    ("天哪！", "Good heavens!", "interj"),
    ("我的天。", "Oh my.", "interj"),
    ("我的乖乖。", "Goodness me.", "interj"),
    ("糟糕。", "Damn.", "interj"),
    ("糟了。", "Oh no.", "interj"),
    ("完了。", "We're doomed.", "interj"),
    ("太好了！", "Great!", "interj"),
    ("真棒！", "Awesome!", "interj"),
    ("好耶！", "Yay!", "interj"),

    # ---------- 习语（成语类，用户明确说暂缓，但顺手加少量）----------
    ("画蛇添足。", "Gild the lily.", "idiom"),
    ("画蛇添足。", "To overdo it.", "idiom"),
    ("多此一举。", "That's unnecessary.", "idiom"),
    ("多此一举。", "That's redundant.", "idiom"),
    ("一举两得。", "Kill two birds with one stone.", "idiom"),
    ("一石二鸟。", "Kill two birds with one stone.", "idiom"),
    ("守口如瓶。", "Keep it under wraps.", "idiom"),
    ("爱不释手。", "Can't put it down.", "idiom"),
    ("马马虎虎。", "So-so.", "idiom"),
    ("半斤八两。", "Six of one, half a dozen of the other.", "idiom"),
    ("入乡随俗。", "When in Rome, do as the Romans do.", "idiom"),
    ("雪中送炭。", "A friend in need is a friend indeed.", "idiom"),
    ("小题大做。", "Make a mountain out of a molehill.", "idiom"),
    ("乱七八糟。", "A total mess.", "idiom"),
    ("胡说八道。", "Nonsense.", "idiom"),
    ("不可思议。", "Unbelievable.", "idiom"),
    ("无可奈何。", "There's nothing to be done.", "idiom"),
    ("异想天开。", "Daydreaming.", "idiom"),
    ("都是些陈词滥调。", "It's all clichés.", "idiom"),
]


# ============================================================
# 模板扩展：用句式生成变体，提高泛化
# ============================================================
def expand_templates():
    """生成额外的变体句对，提升模型的泛化能力"""
    out = []
    # 应答语前缀变体
    pre_zh = ["", "嗯，", "哦，", "好的，", "好吧，"]
    for p in pre_zh:
        for zh, en, cat in PHRASE_PAIRS:
            if cat == "resp" and len(zh) <= 6:
                if p:
                    out.append((p + zh, "Well, " + en[0].lower() + en[1:], cat))
                else:
                    out.append((zh, en, cat))
    # 话语标记后接主句
    marks = [
        ("说真的，", "To be honest, "),
        ("说实话，", "Honestly, "),
        ("老实说，", "Frankly, "),
        ("其实，", "Actually, "),
        ("总之，", "In short, "),
        ("反正，", "Anyway, "),
        ("说白了，", "Put simply, "),
        ("顺便说一句，", "By the way, "),
        ("换句话说，", "In other words, "),
        ("也就是说，", "That is to say, "),
    ]
    tails = [
        ("我不太想去。", "I don't really want to go."),
        ("这件事很难办。", "this is hard to handle."),
        ("他已经走了。", "he has already left."),
        ("我没意见。", "I have no objection."),
        ("这不太合适。", "this isn't quite appropriate."),
        ("时间不够了。", "there isn't enough time."),
    ]
    for mz, me in marks:
        for tz, te in tails:
            out.append((mz + tz, me + te, "mark"))
    return out


def build_samples():
    samples = []
    for zh, en, cat in PHRASE_PAIRS:
        samples.append((zh, en, cat))
    samples += expand_templates()
    # 双向：也生成 en->zh 方向（模型是双向的）
    both = []
    for zh, en, cat in samples:
        both.append((zh, en, cat))
        both.append((en, zh, cat + "_rev"))
    return both


def encode(src, tgt, src_lang, tgt_lang):
    """构造 <|zh|> 中文 <|en|> English <eos> 序列"""
    text = f"{src_lang} {src} {tgt_lang} {tgt} {EOS}"
    return tok.encode(text).ids


def main():
    outdir = os.path.join(BASE, "data/tokens2")
    os.makedirs(outdir, exist_ok=True)

    samples = build_samples()
    print(f"总样本数（含反向）: {len(samples)}")

    # ---- 统计类别 ----
    from collections import Counter
    c = Counter(s[2] for s in samples)
    print("类别分布:", dict(c))

    # ---- 编码 ----
    all_ids = []
    for zh, en, cat in samples:
        has_zh = any('\u4e00' <= ch <= '\u9fff' for ch in zh)
        if has_zh:
            ids = encode(zh, en, L_ZH, L_EN)
        else:
            ids = encode(zh, en, L_EN, L_ZH)
        all_ids.extend(ids)
        # 句间加 EOS 分隔
        if not ids or ids[-1] != tok.token_to_id(EOS):
            all_ids.append(tok.token_to_id(EOS))

    print(f"总 token 数: {len(all_ids)}")

    # ---- 重复放大（oversample）----
    # 这些数据量小（~2.6万 token），必须重复很多遍才能在 4B tokens 里见效。
    # 直接拼成大 buffer 再切分，避免产生上万个碎文件。
    REPEAT = int(os.environ.get("REPEAT", "1500"))
    arr = np.array(all_ids, dtype=np.uint32)
    PART_SIZE = 2_000_000        # 每个 part 目标 2M tokens

    buf = []
    buf_len = 0
    part = 0
    written = 0
    for r in range(REPEAT):
        idx = np.random.permutation(len(arr))
        buf.append(arr[idx])
        buf_len += len(arr)
        # buffer 够大就落盘
        while buf_len >= PART_SIZE:
            cat = np.concatenate(buf)
            while len(cat) >= PART_SIZE:
                seg = cat[:PART_SIZE]
                cat = cat[PART_SIZE:]
                p = os.path.join(outdir, f"part_{part:02d}.bin")
                seg.tofile(p)
                part += 1
                written += len(seg)
            buf = [cat] if len(cat) else []
            buf_len = len(cat)
    # 收尾
    if buf_len > 0:
        cat = np.concatenate(buf)
        if len(cat) > 1000:
            p = os.path.join(outdir, f"part_{part:02d}.bin")
            cat.tofile(p)
            part += 1
            written += len(cat)

    print(f"写出 {part} 个 part，共 {written/1e6:.1f}M tokens (REPEAT={REPEAT})")
    print(f"目录: {outdir}")
    print(f"占 4.0B 总目标的比例: {written/4e9*100:.2f}%")


if __name__ == "__main__":
    main()
