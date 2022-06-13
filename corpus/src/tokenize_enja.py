import re
import sacremoses as sm
import unicodedata
import MeCab
import unidic


def tokenize_en(en):
    mt = sm.MosesTokenizer(lang='en')
    en = unicodedata.normalize('NFKC', en)
    en = re.sub(mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
    en = mt.tokenize(en, escape=False)
    en = ' '.join(en)
    en = en.lower()
    return en


def tokenize_ja(ja):
    wakati = MeCab.Tagger('-Owakati')
    ja = unicodedata.normalize('NFKC', ja)
    ja = wakati.parse(ja)
    return ja

if __name__=="__main__":
    ja = "くるまでまつ"
    res = tokenize_ja(ja)
    print(res)
