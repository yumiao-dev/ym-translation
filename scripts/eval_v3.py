# -*- coding: utf-8 -*-
"""ym-Translation 评估 v3：语气词 / 情态 / 口语应答 专项（不含成语）

设计原则：
  1. 全部为**日常口语高频句**，且机翻（Google/DeepSeek/OPPO键盘）常出错或丢语气
  2. 每类给出参考译文（ref），便于人工对照，不自动算分（BLEU 对短句噪声大）
  3. 明确排除成语、谚语
"""
import os, sys, time, argparse, glob
import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/mnt/workspace/ym-Translation/scripts")
import rmsnorm_shim  # noqa: F401  torch<2.4 的 nn.RMSNorm 兼容补丁
from train_trans import YMTranslate, CTX

BASE = "/mnt/workspace/ym-Translation"
_c = glob.glob(os.path.join(BASE, "**/tokenizer.json"), recursive=True)
TOK_PATH = _c[0] if _c else os.path.join(BASE, "tokenizer/tokenizer.json")
L_ZH, L_EN = "<|zh|>", "<|en|>"
STOP_MARKS = ["<|eos|>", "<eos>", "<|endoftext|>"]


def load(ckpt):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = YMTranslate().to(dev)
    d = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = d["model"]
    if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    return m, dev, d.get("step", -1)


@torch.no_grad()
def gen(model, tok, dev, text, direction, max_new=160):
    prompt = f"{L_ZH} {text} {L_EN}" if direction == "zh2en" else f"{L_EN} {text} {L_ZH}"
    ids = tok.encode(prompt).ids[-2048:]
    cur = torch.tensor([ids], dtype=torch.long, device=dev)
    out = list(ids)
    seg = ""
    t0 = time.time()
    for _ in range(max_new):
        logits, _ = model(cur[:, -CTX:])
        tid = int(torch.argmax(logits[0, -1]))
        out.append(tid)
        cur = torch.cat([cur, torch.tensor([[tid]], device=dev)], dim=1)
        seg = tok.decode(out[len(ids):])
        hit = None
        for mk in STOP_MARKS:
            p = seg.find(mk)
            if p >= 0 and (hit is None or p < hit[0]):
                hit = (p, mk)
        for mk in (L_ZH, L_EN):
            p = seg.find(mk)
            if p > 0 and (hit is None or p < hit[0]):
                hit = (p, mk)
        if hit:
            seg = seg[:hit[0]]
            break
    dt = time.time() - t0
    return seg.strip(), dt


# (类别, 中文源句, 参考/期望英文, 机翻典型错误说明)
CASES = [
    # ---------- A. 句末语气词 ----------
    ("A句末语气词", "你是学生吗？",              "Are you a student?",                "易加 Do you...? 冗余"),
    ("A句末语气词", "你去哪儿呢？",              "Where are you going?",              "易译成 What about you go?"),
    ("A句末语气词", "我们走吧。",                "Let's go.",                          "易译 We go. 丢祈使软化"),
    ("A句末语气词", "这很简单嘛。",              "It's simple, obviously. / Come on, it's easy.", "易丢 obviously 语气"),
    ("A句末语气词", "不想去就别去呗。",           "If you don't want to go, just don't.",  "易译 If you don't want to go, then don't go (语气生硬)"),
    ("A句末语气词", "好漂亮啊！",                "Oh, how beautiful!",                 "易译 So beautiful! 丢感叹词"),
    ("A句末语气词", "你快点呀！",                "Hurry up!",                          "易译 You hurry up 中式"),

    # ---------- B. 体标记 / 时态 ----------
    ("B体标记",   "我吃了。",                  "I have eaten. / I've eaten.",         "易译 I eat. 丢完成体"),
    ("B体标记",   "他去过北京。",               "He has been to Beijing.",             "易混 has gone to"),
    ("B体标记",   "他在吃饭呢。",               "He is eating (right now).",           "易丢进行体/呢"),
    ("B体标记",   "我明天就去。",               "I'll go tomorrow.",                   "易译 I go tomorrow 时态错"),
    ("B体标记",   "还好我带了伞。",              "Good thing I brought an umbrella.",   "易译 Fortunately... 尚可"),

    # ---------- C. 口语应答语 ----------
    ("C应答语",   "好的。",                   "Okay. / Sure.",                       "易译 Good."),
    ("C应答语",   "我知道了。",                "I see. / Got it.",                    "❌易译 I know.（语义错：我已知道 vs 我明白了）"),
    ("C应答语",   "我知道了，你不用说了。",       "I get it, you don't have to say it.", "同上，I know 错"),
    ("C应答语",   "收到。",                   "Got it. / Roger that.",               "易译 Received 生硬"),
    ("C应答语",   "嗯嗯。",                   "Uh-huh. / Yeah.",                     "易音译 En en"),
    ("C应答语",   "嗯？",                     "Huh? / What?",                        "易丢"),
    ("C应答语",   "嗯哼。",                   "Uh-huh.",                             "易丢"),
    ("C应答语",   "算了吧。",                  "Never mind. / Forget it.",            "易译 Forget about it 尚可"),
    ("C应答语",   "随你便。",                  "Suit yourself. / Whatever.",           "❌易译 As you like（中式）"),
    ("C应答语",   "看着办吧。",                "Play it by ear. / You figure it out.", "❌易译 Look and do it（完全崩）"),
    ("C应答语",   "辛苦了。",                  "Thanks for your hard work.",           "❌易译 You're tired / Hard work（中式）"),
    ("C应答语",   "哪里哪里。",                "You flatter me. / Not at all.",        "❌易译 Where where（经典崩）"),

    # ---------- D. 感叹词 / 拟声 ----------
    ("D感叹词",   "哎呀，我忘了带钥匙。",        "Oh no, I forgot my keys.",            "易丢 Oh no"),
    ("D感叹词",   "哎哟，好疼！",               "Ouch, that hurts!",                   "易译 Oh, it hurts"),
    ("D感叹词",   "哇，太棒了！",               "Wow, that's great!",                  "易丢 Wow"),
    ("D感叹词",   "唉，随便吧。",               "Sigh, whatever.",                     "易丢叹息"),
    ("D感叹词",   "哼，我才不信呢。",            "Hmph, I don't believe it.",           "易丢 Hmph"),

    # ---------- E. 话语标记 / 口头禅 ----------
    ("E话语标记", "说真的，我不太想去。",         "To be honest, I don't really want to go.", "易丢 To be honest"),
    ("E话语标记", "那个……我是什么意思呢？",      "Well... what do I mean?",             "❌易直译 That...（崩）"),
    ("E话语标记", "反正我不管了。",              "Anyway, I'm done with this.",         "易丢 Anyway"),
    ("E话语标记", "说白了就是钱的问题。",         "Put simply, it's a money problem.",    "易译 Say it plainly 中式"),
    ("E话语标记", "不是我说你，你这样不行。",      "No offense, but this won't work.",     "❌易直译 Not me saying you（崩）"),
    ("E话语标记", "真是的，又迟到了。",            "Honestly, late again.",               "易丢抱怨语气"),
    ("E话语标记", "别这样嘛，好不好？",           "Come on, please?",                    "易译 Don't be like this, ok（生硬）"),
    ("E话语标记", "差不多得了。",                "That's enough. / Cut it out.",         "❌易译 Almost done（反向错）"),
    ("E话语标记", "无所谓啦。",                  "It doesn't matter.",                  "易丢啦 轻松语气"),

    # ---------- F. 反语 / 委婉 / 敬语 ----------
    ("F委婉",    "劳驾，借过一下。",             "Excuse me, could I get through?",      "易译 Excuse me, borrow a pass（崩）"),
    ("F委婉",    "要不……你还是别去了？",         "Maybe... you'd better not go?",        "易丢委婉"),
    ("F委婉",    "恕我直言，这不太合适。",        "With all due respect, this isn't appropriate.", "易丢 With all due respect"),
    ("F委婉",    "我可不可以这么理解？",          "Can I take it that way?",             "易译 Can I understand like this（尚可）"),
    ("F委婉",    "麻烦你帮我个忙。",             "Could you do me a favor?",            "易译 Please help me do a favor（中式）"),
    ("F委婉",    "那就打扰了。",                "Sorry to bother you, then.",          "易丢语境"),

    # ---------- G. 对照组：直白陈述（模型应做得好） ----------
    ("G对照",    "他昨天去了北京。",             "He went to Beijing yesterday.",        "基线"),
    ("G对照",    "我正在学习机器翻译。",          "I am learning machine translation.",   "基线"),
    ("G对照",    "请把这本书给我。",             "Please give me this book.",            "基线"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "output/ckpt/step_003814.pt"))
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=len(CASES))
    ap.add_argument("--ref", action="store_true", help="同时打印参考译文")
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOK_PATH)
    model, dev, step = load(args.ckpt)
    print(f"# ckpt_step={step} dev={dev} n_cases={len(CASES)}")
    for i, (cat, src, ref, note) in enumerate(CASES, 1):
        if i <= args.lo or i > args.hi:
            continue
        has_zh = any('\u4e00' <= ch <= '\u9fff' for ch in src)
        d = "zh2en" if has_zh else "en2zh"
        hyp, dt = gen(model, tok, dev, src, d)
        print(f"[{i:02d}] {cat} | {src}  ==>  {hyp}   ({dt:.2f}s)")
        if args.ref:
            print(f"      REF: {ref}")
            print(f"      NOTE: {note}")


if __name__ == "__main__":
    main()
