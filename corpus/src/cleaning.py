import unicodedata
from tqdm import tqdm
import re
import regex


def is_en(sents):
    res = []
    en = re.compile("""[a-zA-Z   # アルファベット
                        0-9      # 数字
                        \u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E   # ASCII記号の半角版
    ]+""")

    print("\nChecking if downloaded sentences are truly English sentences or not...")
    for sent in tqdm(sents):
        if en.fullmatch(sent) is None:
            res.append(False)
        else:
            res.append(True)
    return res


def is_ja(sents):
    res = []
    ja = regex.compile("""[\u3041-\u309F                    # ひらがな
                            \u30A1-\u30FF\uFF66-\uFF9F      # カタカナ
                            0-9０-９                        # アラビア数字
                            \p{Numeric_Type=Numeric}        # 漢数字、ローマ数字
                            \p{Script_Extensions=Han}       # 漢字
                            \u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E   # ASCII文字(記号)の半角版
                            \uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F   # ASCII文字(記号)全角版と日本語の記号の半角版
    ]+""")

    print("\nChecking if downloaded sentences are truly Japanese sentences or not...")
    for sent in tqdm(sents):
        if ja.fullmatch(sent) is None:
            res.append(False)
        else:
            res.append(True)
    return res


def cleaning(en_sents, ja_sents):
    """
    正規表現を用いてデータセットに含まれるノイズ(記号, URL, メールアドレス, 日英以外の言語の文, etc...)を除去する関数
    高速化のため、正規表現のパターンを事前にコンパイルしておく。

    同じ機能を実現するための正規表現のパターンは一通りではなく、いくつも考えられる。
    しかし、パターンによってはプログラムを停止させてしまうことがあるから、
    新しい機能をこの関数に追加するときは、十分にテストする。
    """
    brackets = re.compile(r"""\<.*?\>|\{.*?\}|\(.*?\)|\[.*?\]|   # 括弧（半角）
                           [\u3008-\u3011\u3014-\u301B]+.*?[\u3008-\u3011\u3014-\u301B]+   # 括弧（全角）
    """)
    unwanted = re.compile(r"[*#^「」『 』]+")
    msc = re.compile(r"\\\\|\t|\\\\t|\r|\\\\r")
    newlines = re.compile(r"\\\n|\n")
    urls = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    email = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    encoding_err = re.compile("0000,0000,0000,\w*?")
    multi_space = re.compile("[ 　]{2,}")
    emoji = regex.compile("\p{Emoji_Presentation=Yes}+")

    cleaned_en, cleaned_ja = [], []
    length = min(len(en_sents), len(ja_sents))
    print("\nCleaning sentences...")

    for en_sent, ja_sent in tqdm(zip(en_sents, ja_sents), total=length):
        en_sent = unicodedata.normalize("NFKC", en_sent).strip()
        ja_sent = unicodedata.normalize("NFKC", ja_sent).strip()
        en_sent = urls.sub('', en_sent)
        ja_sent = urls.sub('', ja_sent)

        en_sent = email.sub('', en_sent)
        ja_sent = email.sub('', ja_sent)

        en_sent = msc.sub('', en_sent)
        ja_sent = msc.sub('', ja_sent)

        en_sent = newlines.sub('', en_sent)
        ja_sent = newlines.sub('', ja_sent)

        en_sent = emoji.sub('', en_sent)
        ja_sent = emoji.sub('', ja_sent)

        en_sent = brackets.sub('', en_sent)
        ja_sent = brackets.sub('', ja_sent)

        en_sent = unwanted.sub('', en_sent)
        ja_sent = unwanted.sub('', ja_sent)

        en_sent = multi_space.sub('', en_sent)
        ja_sent = multi_space.sub('', ja_sent)

        en_sent = encoding_err.sub('', en_sent)
        ja_sent = encoding_err.sub('', ja_sent)

        cleaned_en.append(en_sent.strip())
        cleaned_ja.append(ja_sent.strip())

    en_tf_ls = is_en(cleaned_en)
    ja_tf_ls = is_ja(cleaned_ja)

    en_ls, ja_ls = [], []
    for idx, (en_tf, ja_tf) in enumerate(zip(en_tf_ls, ja_tf_ls)):
        if en_tf == True and ja_tf == True:
            en_ls.append(cleaned_en[idx])
            ja_ls.append(cleaned_ja[idx])

    return en_ls, ja_ls


# テスト用コード
if __name__ == "__main__":
    en_sents = [
        "the college 🤩provided courses in science , engineering and art , and also established its own internal degree courses in science and engineering , which were ratified by the university of london .",
        "he won the \t fossati prize . \n https://en.wikipedia.org/wiki/Giampiero_Fossati",
        "he was born in haarlem as the son of aart jansz and became a lawyer .",
        "He said (私は日系アメリカ人二世です。).",
        "I    have a 💯(pen). timo.33@gmail.com"
    ]

    ja_sents = [
        "当時 の 😀😃学校 は 科学 、 工学 、 🤩芸術 の コース を 開講 し 、 それ ら の コース は ロンドン 大学 より 学位 の 承認 を 受け て い た 。",
        "^ ピタリMHz 賞 を 獲得 し た 。http://en.wikipedia.org/wiki/Giampiero_Fossati",
        "er wurde in haarlem als sohn von aart jansz geboren und wurde rechtsanwalt.",
        "彼は「日系アメリカ人二世です。０IV」と言った。",
        "私はペン　　を持っています。*💯"
    ]

    #en_tf_ls = is_en(en_sents)
    #ja_tf_ls = is_ja(ja_sents)

    #en_ls, ja_ls = [], []
    # for idx, (en_tf, ja_tf) in enumerate(zip(en_tf_ls, ja_tf_ls)):
    #    if en_tf == True and ja_tf == True:
    #        en_ls.append(en_sents[idx])
    #        ja_ls.append(ja_sents[idx])

    en_ls, ja_ls = cleaning(en_sents=en_sents, ja_sents=ja_sents)

    print(en_ls)
    print(ja_ls)
