import dl_tatoeba as dl
import tokenize_enja as tk
import filter as fl
import split_dataset as spl
import tqdm as t
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument("--repo_path", type=str,
                        help="absolute path of Machine_Translation_Proto repository")
    args = parser.parse_args()
    repo_path = args.repo_path

    # データセットをダウンロードしてリスト化する
    dl.dl_tatoeba(repo_path)
    en_ls, ja_ls = dl.json2list(repo_path)

    # 英文と日本文をそれぞれトークン化する
    print("\nTokenizing English sentences...")
    en_ls = [tk.tokenize_en(en) for en in t.tqdm(en_ls)]
    print("\nTokenizing Japanese sentences...")
    ja_ls = [tk.tokenize_ja(ja) for ja in t.tqdm(ja_ls)]

    # フィルタリング
    en_ls, ja_ls = fl.len_filter(en_ls, ja_ls, 4, 16, truncate=True)
    en_ls, ja_ls = fl.overlap_filter(en_ls, ja_ls)
    en_ls, ja_ls = fl.ratio_filter(en_ls, ja_ls)

    split_ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}
    spl.split_dataset(en_ls, ja_ls, split_ratio, repo_path)
