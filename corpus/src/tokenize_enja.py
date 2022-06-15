import re
import sacremoses as sm
import unicodedata
import MeCab
import threading
from typing import List
from tqdm import tqdm


class Tokenization():
    def __init__(self, lang, threading=False, num_threads=2):
        self.lang = lang
        self.threading = threading
        if lang == "en":
            self.tokenizer = sm.MosesTokenizer(lang='en')
        elif lang == "ja":
            self.tokenizer = MeCab.Tagger("-Owakati")
        else:
            raise ValueError(
                "Error: Language %s is not supported." % self.lang)
        self.num_threads = num_threads
        self.out = []   # 各部分リストをトークン化した時の結果をまとめて保持するためのリスト

    def tokenize_en(self, en_sents: List[str]):
        en_ls = []
        for en in en_sents:
            en = unicodedata.normalize("NFKC", en)
            en = re.sub(
                self.tokenizer.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
            en = self.tokenizer.tokenize(en, escape=False)
            en = ' '.join(en).lower()
            en_ls.append(en)
        self.out.append(en_ls)

    def tokenize_ja(self, ja_sents: List[str]):
        ja_ls = []
        for ja in ja_sents:
            ja = unicodedata.normalize("NFKC", ja)
            ja = self.tokenizer.parse(ja)
            ja_ls.append(ja)
        self.out.append(ja_ls)

    def tokenize(self, sents: List[str]):
        num_sents = len(sents)
        size = int(num_sents / self.num_threads)
        threads = []
        tgt = self.tokenize_en if self.lang == "en" else self.tokenize_ja
        if self.threading:
            for i in range(self.num_threads):
                head = i * size
                tail = (i+1) * size if i != (self.num_threads-1) else -1
                thread = threading.Thread(target=tgt, args=[sents[head: tail]])
                threads.append(thread)
                thread.start()

            for thread in tqdm(threads):
                thread.join()
            tgt([sents[-1]])
        else:
            tgt(sents)

        res = [sent for sents in self.out for sent in sents]
        return res


if __name__ == "__main__":
    ja_ls = ["何かしてみましょう。",
             "私は眠らなければなりません。",
             "そろそろ寝なくちゃ。",
             "今日は６月１８日で、ムーリエルの誕生日です！",
             "ムーリエルは２０歳になりました。",
             "パスワードは「Muiriel」です。",
             "まもなく私は戻って来ます。",
             "私は言葉に詰まった。"
             ]

    t = Tokenization(lang="ja", threading=True)
    out = t.tokenize(ja_ls)
    print(''.join(out))
