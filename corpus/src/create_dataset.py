import dl_tatoeba as tatoeba
import filter as fl
import split_dataset as spl
import argparse
import dl_WikiMatrix as wiki
import tokenize_enja as tkn
import sys
import time
from cleaning import cleaning


def print_sents(sents):
    for idx, sent in enumerate(sents):
        print("{}:   {}".format(idx, sent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='usage')
    parser.add_argument("--repo_path", type=str,
                        help="absolute path of Machine_Translation_Proto repository")
    parser.add_argument("--tatoeba", action="store_true",
                        help="use Tatoeba dataset")
    parser.add_argument("--WikiMatrix", action="store_true",
                        help="use WikiMatrix dataset.")
    parser.add_argument("--len_filter", action="store_true",
                        help="turn on/off the length filter")
    parser.add_argument("--min_len", type=int, default=4,
                        help="valid minimum length of sentences in a dataset")
    parser.add_argument("--max_len", type=int, default=256,
                        help="valid maximum length of sentences in a dataset")
    parser.add_argument("--overlap_filter", action="store_true",
                        help="turn on/off the length filter")
    parser.add_argument("--ratio_filter", action="store_true",
                        help="turn on/off the ratio filter")
    parser.add_argument("--freq_filter", action="store_true",
                        help="turn on/off the freq filter")
    parser.add_argument("--freq_thld", type=int, default=2,
                        help="threshold for filtering words by frequency")
    parser.add_argument("--multiproc", action="store_true",
                        help="turn on/off multi-processing to accelerate tokenization and filtering by frequency")
    parser.add_argument("--num_procs_tkn", type=int, default=8,
                        help="the number of processes to accelerate tokenization\n Default: 8   Valid range: 1 <= num_procs_tkn <= 16")
    parser.add_argument("--num_procs_freq", type=int, default=4,
                        help="the number of processes to accelerate creating frequency dictionaries\n Default: 4   Valid range: 1 <= num_procs_tkn <= 8")

    args = parser.parse_args()
    repo_path = args.repo_path

    en_tmp_ls, ja_tmp_ls = [], []
    # Tatoebaデータセットをダウンロードしてリスト化する
    if args.tatoeba:
        tatoeba.dl_tatoeba(repo_path)
        tatoeba_en, tatoeba_ja = tatoeba.json2list(repo_path)
        en_tmp_ls.append(tatoeba_en)
        ja_tmp_ls.append(tatoeba_ja)

    # WikiMatrixデータセットをダウンロードしてリスト化する
    if args.WikiMatrix:
        wiki_en, wiki_ja = wiki.dl_WikiMatrix(repo_path)
        en_tmp_ls.append(wiki_en)
        ja_tmp_ls.append(wiki_ja)

    if len(en_tmp_ls) == 0 or len(ja_tmp_ls) == 0:
        print("You need to specify at least one dataset to create a new dataset.")
        sys.exit()
    else:
        en_ls = [en_sent for en_sents in en_tmp_ls for en_sent in en_sents]
        ja_ls = [ja_sent for ja_sents in ja_tmp_ls for ja_sent in ja_sents]

    en_ls, ja_ls = cleaning(en_ls, ja_ls)

    # 英文と日本文をそれぞれトークン化する
    num_procs_tkn = args.num_procs_tkn
    if num_procs_tkn < 1 or num_procs_tkn > 16:
        print("The value num_procs_tkn %d is invalid. " % num_procs_tkn)
        print("It is replaced by %d." % 8)
        num_procs_tkn = 8

    print("\nTokenizing sentences...")
    start = time.time()
    tkn = tkn.Tokenization(multiproc=args.multiproc,
                           num_procs=num_procs_tkn)
    en_ls, ja_ls = tkn.tokenize(en_ls, ja_ls)
    end = time.time()
    print("%d seconds for tokenizing sentences" % int(end - start))

    # フィルタリング
    if args.len_filter:
        min = args.min_len
        max = args.max_len
        if min < 1 or min > 16:
            print(
                "The minimum length of sentences should be in a range: 1 <= min_len <= 16")
            print("Specified min_len %d is replaced by %d" % (min, 4))
            min = 4
        if max < 16 or max > 256:
            print(
                "The maximum length of sentences should be in a range: 16 <= min_len <= 256")
            print("Specified max_len %d is replaced by %d" % (max, 32))
            max = 32
        en_ls, ja_ls = fl.len_filter(en_ls, ja_ls, min, max, truncate=True)
    if args.overlap_filter:
        en_ls, ja_ls = fl.overlap_filter(en_ls, ja_ls)
    if args.ratio_filter:
        en_ls, ja_ls = fl.ratio_filter(en_ls, ja_ls)
    if args.freq_filter:
        en_ls, ja_ls = fl.freq_filter(
            en_ls, ja_ls, args.freq_thld, args.multiproc, args.num_procs_freq, sort_fd=False)

    split_ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}
    spl.split_dataset(en_ls, ja_ls, split_ratio, repo_path)
