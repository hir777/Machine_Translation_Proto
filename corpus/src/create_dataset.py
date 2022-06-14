from email.policy import default
import dl_tatoeba as dl
import tokenize_enja as tk
import filter as fl
import split_dataset as spl
import tqdm as t
import argparse
import dl_WikiMatrix as wiki


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument("--repo_path", type=str,
                        help="absolute path of Machine_Translation_Proto repository")
    parser.add_argument("--len_filter", action="store_true",
                        help="turn on/off the length filter")
    parser.add_argument("--overlap_filter", action="store_true",
                        help="turn on/off the length filter")
    parser.add_argument("--ratio_filter", action="store_true",
                        help="turn on/off the ratio filter")
    parser.add_argument("--freq_filter", action="store_true",
                        help="turn on/off the freq filter")
    parser.add_argument("--freq_thld", type=int, default=4,
                        help="threshold for filtering words by frequency")

    args = parser.parse_args()
    repo_path = args.repo_path

    # Tatoebaデータセットをダウンロードしてリスト化する
    dl.dl_tatoeba(repo_path)
    tatoeba_en_ls, tatoeba_ja_ls = dl.json2list(repo_path)

    # WikiMatrixデータセットをダウンロードしてリスト化する
    wiki_en_ls, wiki_ja_ls = wiki.dl_WikiMatrix(repo_path)

    en_ls = tatoeba_en_ls + wiki_en_ls
    ja_ls = tatoeba_ja_ls + wiki_ja_ls
    # 英文と日本文をそれぞれトークン化する
    print("\nTokenizing English sentences...")
    en_ls = [tk.tokenize_en(en) for en in t.tqdm(en_ls)]
    print("\nTokenizing Japanese sentences...")
    ja_ls = [tk.tokenize_ja(ja) for ja in t.tqdm(ja_ls)]

    # フィルタリング
    if args.len_filter:
        en_ls, ja_ls = fl.len_filter(en_ls, ja_ls, 3, 256, truncate=True)
    if args.overlap_filter:
        en_ls, ja_ls = fl.overlap_filter(en_ls, ja_ls)
    if args.ratio_filter:
        en_ls, ja_ls = fl.ratio_filter(en_ls, ja_ls)
    if args.freq_filter:
        en_ls, ja_ls = fl.freq_filter(en_ls, ja_ls, args.freq_thld)

    split_ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}
    spl.split_dataset(en_ls, ja_ls, split_ratio, repo_path)
