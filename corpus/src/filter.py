from collections import defaultdict
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

    print("\nFiltering by ratio...")
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


def get_freq_dict(en_sents, ja_sents):
    en_dict, ja_dict = {}, {}
    print("\nCreating frequency lists...")
    for en_sent, ja_sent in t.tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        en_sent = en_sent.strip().split()
        ja_sent = ja_sent.strip().split()
        for en in en_sent:
            if en in en_dict:
                en_dict[en] += 1
            else:
                en_dict[en] = 1

        for ja in ja_sent:
            if ja in ja_dict:
                ja_dict[ja] += 1
            else:
                ja_dict[ja] = 1

    return en_dict, ja_dict


def sort_freq_dict(en_dict, ja_dict):
    en_freq, ja_freq = list(en_dict.items()), list(ja_dict.items())
    en_freq.sort(key=lambda x: -x[1])
    ja_freq.sort(key=lambda x: -x[1])
    en_freq = {freq[0]: freq[1] for freq in en_freq}
    ja_freq = {freq[0]: freq[1] for freq in ja_freq}
    return en_freq, ja_freq


def replace_by_unk(sent, freq_dict, freq_thld):
    w_ls = [w if freq_dict[w] >=
            freq_thld else "<unk>" for w in sent.strip().split()]
    return ' '.join(w_ls)


def freq_filter(en_sents, ja_sents, freq_thld):
    en_ls, ja_ls = [], []
    en_freq, ja_freq = get_freq_dict(en_sents, ja_sents)
    en_freq, ja_freq = sort_freq_dict(en_freq, ja_freq)
    print("\nFiltering by frequency...")
    en_ls = [replace_by_unk(en_sent, en_freq, freq_thld)
             for en_sent in t.tqdm(en_sents)]
    ja_ls = [replace_by_unk(ja_sent, ja_freq, freq_thld)
             for ja_sent in t.tqdm(ja_sents)]
    return en_ls, ja_ls


if __name__ == "__main__":
    en_ls = ["I have a pen .", "I have a pen .", "I am a dog lover"]
    ja_ls = ["私 は ペン を 持って いる 。", "私 は 鉛筆 を 持って いる 。", "私 は 愛犬家 です 。"]
    en_ls, ja_ls = overlap_filter(en_ls, ja_ls)
    en_ls, ja_ls = freq_filter(en_ls, ja_ls, 3)
    print(en_ls)
    print(ja_ls)
