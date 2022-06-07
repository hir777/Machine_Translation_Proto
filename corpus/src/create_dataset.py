import os
from tqdm import tqdm
from dl_tatoeba import dl_tatoeba
from dl_tatoeba import json2list
from tokenize import tokenize_en
from tokenize import tokenize_ja
import filter as fl
from split_dataset import split_dataset

if __name__ == "__main__":
    repo_path = "/home/hiroshi/tmp/Machine_Translation_Proto/"
    # データセットをダウンロードしてリスト化する
    dl_tatoeba(repo_path)
    en_ls, ja_ls = json2list(repo_path)

    # 英文と日本文をそれぞれトークン化する
    en_ls = [tokenize_en(en) for en in tqdm(en_ls)]
    ja_ls = [tokenize_ja(ja) for ja in tqdm(ja_ls)]

    # フィルタリング
    en_ls, ja_ls = fl.len_filter(en_ls, ja_ls, 4, 16, truncate=True)
    en_ls, ja_ls = fl.overlap_filter(en_ls, ja_ls)
    en_ls, ja_ls = fl.ratio_filter(en_ls, ja_ls)

    split_ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}
    split_dataset(en_ls, ja_ls, split_ratio, repo_path)
