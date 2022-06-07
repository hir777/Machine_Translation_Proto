import re
import sacremoses as sm
import unicodedata
import spacy


def tokenize_en(en):
    mt = sm.MosesTokenizer(lang='en')
    en = unicodedata.normalize('NFKC', en)
    en = re.sub(mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
    en = mt.tokenize(en, escape=False)
    en = ' '.join(en)
    en = en.lower()
    return en


def tokenize_ja(ja):
    gt = spacy.load('ja_ginza')
    ja = unicodedata.normalize('NFKC', ja)
    ja = gt.tokenize(ja)
    ja = [token.text for token in ja]
    ja = ' '.join(ja)
    return ja
