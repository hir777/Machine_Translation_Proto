import random as rd
from regex import F
from tqdm import tqdm
from typing import Dict
import numpy as np


def check_ratio(split_ratio: Dict[str, float]):
    ratio = split_ratio.items()
    vals = [r[1] for r in ratio]
    sum = np.sum(vals)

    return False if ( len(ratio) != 3 or not all([val > 0.0 for val in vals]) or sum != 1.0 ) else True

def split_dataset(en_sents, ja_sents, split_ratio: Dict[str, float]):
    train_en, train_ja, dev_en, dev_ja, test_en, test_ja = [], [], [], [], [], []
    en_ja = [en + '\t' + ja for en,
             ja in tqdm(zip(en_sents, ja_sents), total=len(en_sents))]
    rd.shuffle(en_ja)

    size = len(en_ja)
    if not check_ratio:
        return
    train_size = split_ratio["train"] * size
    dev_size = split_ratio["dev"] * size
    train = en_ja[:train_size]
    dev = en_ja[train_size:train_size + dev_size]
    test = en_ja[train_size + dev_size:]

    for bitext in tqdm(train):
        en, ja = bitext.split('\t')
        train_en.append(en)
        train_ja.append(ja)

    for bitext in tqdm(dev):
        en, ja = bitext.split('\t')
        dev_en.append(en)
        dev_ja.append(ja)

    for bitext in tqdm(test):
        en, ja = bitext.split('\t')
        test_en.append(en)
        test_ja.append(ja)

    return train_en, train_ja, dev_en, dev_ja, test_en, test_ja
