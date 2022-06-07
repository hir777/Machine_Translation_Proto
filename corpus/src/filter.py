import tqdm as t
import numpy as np


def lens(s1, s2):
    len_s1 = len(s1.strip().split())
    len_s2 = len(s2.strip().split())
    return len_s1, len_s2


def trunc(x, max):
    x_len = len(x.strip().split())
    return x if x_len <= max else x[:max]


def len_filter(en_sents, ja_sents, min, max, truncate=False):
    en_ls, ja_ls = [], []
    print("\nFiltering by length...")
    for en, ja in t.tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        en_len, ja_len = lens(en, ja)
        if min <= en_len <= max and min <= ja_len <= max:
            en_ls.append(en)
            ja_ls.append(ja)
        elif not (min > en_len or min > ja_len) and truncate:
            en_ls.append(trunc(en, max))
            ja_ls.append(trunc(ja, max))

    return en_ls, ja_ls


def overlap_filter(en_sents, ja_sents):
    en_ls, ja_ls = [], []
    en_dict = {sent: 0 for sent in en_sents}
    ja_dict = {sent: 0 for sent in ja_sents}

    print("\nFiltering by overlap...")
    for en, ja in t.tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        if en_dict[en] == 0 and ja_dict[ja] == 0:
            en_ls.append(en)
            ja_ls.append(ja)
            en_dict[en] += 1
            ja_dict[ja] += 1
        elif (en_dict[en] == 0) ^ (ja_dict[ja] == 0):
            en_ls.append(en)
            ja_ls.append(ja)
            en_dict[en] += 1
            ja_dict[ja] += 1

    return en_ls, ja_ls


def ratio(s1, s2):
    len_s1, len_s2 = lens(s1, s2)
    return len_s1 * 1.0 / len_s2


def ratio_filter(en_sents, ja_sents):
    ratios, en_ls, ja_ls = [], [], []

    print("Filtering by ratio...")
    for en, ja in t.tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        len_en, len_ja = lens(en, ja)
        if len_en == 0 or len_ja == 0:
            continue
        ratios.append(ratio(en, ja))

    # ratioに関する統計情報を計算
    ratios = sorted(ratios)
    N = len(ratios)
    mean = np.mean(ratios)
    std = np.std(ratios)

    for en, ja in t.tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        len_en, len_ja = lens(en, ja)
        if len_en == 0 or len_ja == 0:
            continue
        r = ratio(en, ja)
        if (r < mean - 1.96 * std) or (r > mean + 1.96 * std):
            continue
        en_ls.append(en)
        ja_ls.append(ja)

    return en_ls, ja_ls

if __name__=="__main__":
    en_ls = ["I have a pen .", "I have a pen ."]
    ja_ls = ["私　は　ペン　を　持って　いる　。", "私　は　鉛筆　を　持って　いる　。"]
    en, ja = overlap_filter(en_ls, ja_ls)
