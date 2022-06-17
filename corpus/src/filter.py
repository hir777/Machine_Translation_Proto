from asyncio.format_helpers import _format_callback_source
import queue
from matplotlib.pyplot import get_figlabels
import tqdm as t
import numpy as np
import multiprocessing as mp
import os
import time


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


def get_freq_dict(en_sents, ja_sents, en_queue, ja_queue):
    en_dict, ja_dict = {}, {}
    print("\nCreating frequency lists...: (PID {})".format(os.getpid()))
    for en_sent, ja_sent in t.tqdm(zip(en_sents, ja_sents), total=min(len(en_sents), len(ja_sents))):
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

    en_queue.put(en_dict)
    ja_queue.put(ja_dict)
    print("\nFinished creating frequency lists...: (PID {})".format(os.getpid()))


def sort_freq_dict(en_dict, ja_dict):
    en_freq, ja_freq = list(en_dict.items()), list(ja_dict.items())
    en_freq.sort(key=lambda x: -x[1])
    ja_freq.sort(key=lambda x: -x[1])
    en_freq = {freq[0]: freq[1] for freq in en_freq}
    ja_freq = {freq[0]: freq[1] for freq in ja_freq}
    return en_freq, ja_freq


def concat_freq_dicts(en_queue, ja_queue, multiproc=False, num_procs=4):
    if not multiproc:
        num_procs = 1
    en_freq_ls, ja_freq_ls = [], []
    for _ in range(num_procs):
        en_freq_ls.append(en_queue.get())
        ja_freq_ls.append(ja_queue.get())

    en_freq_dict, ja_freq_dict = {}, {}
    if multiproc:
        for freq_dict in en_freq_ls:
            for key, val in freq_dict.items():
                en_freq_dict[key] = val + \
                    en_freq_dict[key] if key in en_freq_dict else val

        for freq_dict in ja_freq_ls:
            for key, val in freq_dict.items():
                ja_freq_dict[key] = val + \
                    ja_freq_dict[key] if key in ja_freq_dict else val
    else:
        en_freq_dict = en_freq_ls[0]
        ja_freq_dict = ja_freq_ls[0]

    return en_freq_dict, ja_freq_dict


def replace_by_unk(sent, freq_dict, freq_thld):
    w_ls = [w if freq_dict[w] >=
            freq_thld else "<unk>" for w in sent.strip().split()]
    return ' '.join(w_ls)


def freq_filter(en_sents, ja_sents, freq_thld, multiproc=False, num_procs=4, sort_fd=False):
    en_queue = mp.Queue()
    ja_queue = mp.Queue()
    num_sents = min(len(en_sents), len(ja_sents))
    num_procs = num_procs if multiproc and num_procs > 1 else 1
    num_procs = num_procs if num_procs <= num_sents else 2
    size = int(num_sents / num_procs)
    en_ls, ja_ls = [], []

    start = time.time()
    if multiproc:
        tgt_fun = get_freq_dict
        for i in range(num_procs):
            head = i * size
            tail = (i+1) * size if i != (num_procs-1) else num_sents
            proc = mp.Process(target=tgt_fun, args=[
                              en_sents[head:tail], ja_sents[head:tail], en_queue, ja_queue])
            proc.start()
    else:
        get_freq_dict(en_sents[:num_sents],
                      ja_sents[:num_sents], en_queue, ja_queue)
    end = time.time()

    en_freq, ja_freq = concat_freq_dicts(
        en_queue, ja_queue, multiproc, num_procs)
    if sort_fd:
        en_freq, ja_freq = sort_freq_dict(en_freq, ja_freq)
    print("{} seconds for creating a frequency dict".format(end-start))
    print("\nFiltering by frequency...")
    en_ls = [replace_by_unk(en_sent, en_freq, freq_thld)
             for en_sent in t.tqdm(en_sents)]
    ja_ls = [replace_by_unk(ja_sent, ja_freq, freq_thld)
             for ja_sent in t.tqdm(ja_sents)]
    return en_ls, ja_ls


# テストコード
if __name__ == "__main__":
    # フィルター全体のテストコード
    en_ls = ["I have a pen .", "I have a pen .",
             "I am a dog lover .", "I am a bilingual ."]
    ja_ls = ["私 は ペン を 持って いる 。", "私 は 鉛筆 を 持って いる 。",
             "私 は 愛犬家 です 。", "私 は バイリンガル です 。"]
    en_ls, ja_ls = overlap_filter(en_ls, ja_ls)
    en_ls, ja_ls = freq_filter(
        en_ls, ja_ls, freq_thld=2, multiproc=True, num_procs=4, sort_fd=True)
    print(en_ls)
    print(ja_ls)

    # 辞書の結合をテストするためのコード
    #d1 = {"A": 20, "B": 33, "D": 4}
    #d2 = {"A": 32, "B": 44, "C": 5}
    #d3 = {"A": 20, "B": 33, "D": 4}
    #d4 = {"A": 32, "B": 44, "C": 5}
#
    #queue_a = mp.Queue()
    #queue_b = mp.Queue()
    #queue_c = mp.Queue()
    #queue_d = mp.Queue()
#
    # queue_a.put(d1)
    # queue_a.put(d2)
    # queue_b.put(d3)
    # queue_b.put(d4)
#
    # queue_c.put(d1)
    # queue_d.put(d2)
    #a_dict, b_dict = concat_freq_dicts(queue_a, queue_b, True,  2)
    #c_dict, d_dict = concat_freq_dicts(queue_c, queue_d, False, 8)
#
    # print(a_dict)
    # print(b_dict)
    # print(c_dict)
    # print(d_dict)
