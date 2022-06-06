from tqdm import tqdm


def truncate(x, max):
    x_len = len(x.split())
    return x if x_len <= max else x[:max]


def len_filter(en_sents, ja_sents, min, max, truncate=False):
    en_ls, ja_ls = [], []
    for en, ja in tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        en_len = len(en.strip().split())
        ja_len = len(ja.strip().split())
        if min <= en_len <= max and min <= ja_len <= max:
            en_ls.append(en)
            ja_ls.append(ja)
        elif not (min > en_len or min > ja_len) and truncate:
            en_ls.append(truncate(en, max))
            ja_ls.append(truncate(ja, max))

    return en_ls, ja_ls


def overlap_filter(en_sents, ja_sents):
    en_ls, ja_ls = [], []
    en_dict = {sent: 0 for sent in en_sents}
    ja_dict = {sent: 0 for sent in ja_sents}

    for en, ja in tqdm(zip(en_sents, ja_sents), total=len(en_sents)):
        if en_dict[en] == 0 and en_dict[ja] == 0:
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
