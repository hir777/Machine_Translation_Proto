import string
import sys
import unicodedata
from tqdm import tqdm
import re

def is_en(sents):
    res = []
    en = re.compile("""[a-zA-Z   # アルファベット
                            0-9      # 数字
                            \u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E   # ASCII記号の半角版
    ]+""")
    
    for sent in sents:
        if en.fullmatch(sent) is None:
            res.append(False)
        else:
            res.append(True)
    
    return res

def is_ja(sents):
    res = []
    ja = re.compile("""[\u3041-\u309F   # ひらがな
                            \u30A1-\u30FF\uFF66-\uFF9F   # カタカナ
                            \p{Script_Extensions=Han}    # 漢字
                            \p{Numeric_Type=Numeric}0-9０-９ # 数字
                            〇一二三四五六七八九十百千万億兆 # 漢数字
                            \u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E   # ASCII文字(記号)の半角版
                            \uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F   # ASCII文字(記号)全角版と日本語の記号の半角版
    ]+""")

    for sent in sents:
        if ja.fullmatch(sent) is None:
            res.append(False)
        else:
            res.append(True)
    
    return res

def cleaning(en_sents, ja_sents):
    en_brackets = re.compile(r"\<.*?\>|\{.*?\}|\(.*?\)|\[.*?\]")
    ja_brackets = re.compile(
        "[\u3008-\u3011\u3014-\u301B]+.*?[\u3008-\u3011\u3014-\u301B]+")
    unwanted = re.compile(r"[\u*#]")
    encoding_error = re.compile("0000,0000,0000,\w*?")
    multi_space = re.compile("[ ]+")
    msc = re.compile(r"\\\\|\t|\\\\t|\r|\\\\r")
    newlines = re.compile(r"\\\n|\n")
    urls = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    phone_nums = re.compile(
        r"""^(\([0-9]{3}\) ?|[0-9]{3}-)[0-9]{3}-[0-9]{4}$|                # アメリカの電話番号
        ^(?:0|\+?44)\s?(?:\d\s?){9,11}$|                                  # イギリスの電話番号
        [\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}   # 日本の電話番号
        """)
    email = re.compile(
        r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+")
    encoding_error = re.compile("0000,0000,0000,\w*?")
    multi_space = re.compile("[ 　]+")
    emoji = re.compile("\p{Emoji_Presentation=Yes}+")

    cleaned_en, cleaned_ja = [], []
    length = min(len(en_sents), len(ja_sents))
    print("\nCleaning English sentences...")
    for en_sent, ja_sent in tqdm(zip(en_sents, ja_sents), total=length):
        en_sent = unicodedata.normalize("NFKC", en_sent).strip().lower()
        ja_sent = unicodedata.normalize("NFKC", ja_sent).strip().lower()

        en_sent = en_brackets.sub('', en_sent)
        ja_sent = ja_brackets.sub('', ja_sent)

        en_sent = unwanted.sub('', en_sent)

        en_sent = urls.sub('', en_sent)
        ja_sent = urls.sub('', ja_sent)

        en_sent = phone_nums.sub('', en_sent)
        ja_sent = phone_nums.sub('', ja_sent)

        en_sent = email.sub('', en_sent)
        ja_sent = email.sub('', ja_sent)

        en_sent = msc.sub('', en_sent)
        ja_sent = msc.sub('', ja_sent)

        en_sent = newlines.sub('', en_sent)
        ja_sent = newlines.sub('', ja_sent)

        en_sent = emoji.sub('', en_sent)
        ja_sent = emoji.sub('', ja_sent)

        en_sent = multi_space.sub('', en_sent)
        ja_sent = multi_space.sub('', ja_sent)

        en_sent = encoding_error.sub('', en_sent)
        ja_sent = encoding_error.sub('', ja_sent)

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
