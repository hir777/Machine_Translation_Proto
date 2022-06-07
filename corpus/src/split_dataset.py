import random as rd
import tqdm as t
import typing
import numpy as np
import os


def check_ratio(split_ratio: typing.Dict[str, float]):
    ratio = split_ratio.items()
    vals = [r[1] for r in ratio]
    sum = np.sum(vals)
    eps = 0.01

    return False if len(ratio) != 3 or not all([val > 0.0 for val in vals]) or not (1.0 - eps < sum < 1.0 + eps) else True


def write_ds(f_name, f_path, bitexts):
    en_path = os.path.join(f_path, "{}.en".format(f_name))
    ja_path = os.path.join(f_path, "{}.ja".format(f_name))
    with open(en_path, 'w') as f_en, open(ja_path, 'w') as f_ja:
        print("\nWriting {}.en and {}.ja ...".format(f_name, f_name))
        for bitext in t.tqdm(bitexts):
            en, ja = bitext.split('\t')
            f_en.write(en + '\n')
            f_ja.write(ja)


def split_dataset(en_sents, ja_sents, split_ratio: typing.Dict[str, float], repo_path):
    data_path = os.path.join(repo_path, "corpus/data/")
    en_ja = [en + '\t' + ja for en, ja in zip(en_sents, ja_sents)]
    rd.shuffle(en_ja)

    size = len(en_ja)
    if check_ratio:
        train_size = int(split_ratio["train"] * size)
        valid_size = int(split_ratio["valid"] * size)
        train = en_ja[:train_size]
        valid = en_ja[train_size:train_size + valid_size]
        test = en_ja[train_size + valid_size:]

        write_ds('train', data_path, train)
        write_ds('valid', data_path, valid)
        write_ds('test', data_path, test)
