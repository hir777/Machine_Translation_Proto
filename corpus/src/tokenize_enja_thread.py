import re
import sacremoses as sm
import unicodedata
import MeCab
import threading
from typing import List
from tqdm import tqdm


class Tokenization():
    def __init__(self, threading=False, num_threads=2):
        self.threading = threading
        self.en_tokenizer = sm.MosesTokenizer(lang='en')
        self.ja_tokenizer = MeCab.Tagger("-Owakati")
        self.num_threads = num_threads
        self.out = []   # 各部分リストをトークン化した時の結果をまとめて保持するためのリスト(日英)

    def tokenize_en(self, en_sents: List[str]):
        en_ls = []
        for en in en_sents:
            en = unicodedata.normalize("NFKC", en)
            en = re.sub(
                self.en_tokenizer.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
            en = self.en_tokenizer.tokenize(en, escape=False)
            en = ' '.join(en).lower()
            en_ls.append(en)
        return en_ls

    def tokenize_ja(self, ja_sents: List[str]):
        ja_ls = []
        for ja in ja_sents:
            ja = unicodedata.normalize("NFKC", ja)
            ja = self.ja_tokenizer.parse(ja)
            ja_ls.append(ja)
        return ja_ls

    def tokenize_en_ja(self, en_sents: List[str], ja_sents: List[str]):
        en_ls = self.tokenize_en(en_sents)
        ja_ls = self.tokenize_ja(ja_sents)
        self.out.append([en.replace('\t', '') + '\t' + ja.replace('\t', '')
                        for en, ja in zip(en_ls, ja_ls)])

    def tokenize(self, en_sents: List[str], ja_sents: List[str]):
        num_sents = min(len(en_sents), len(ja_sents))
        size = int(num_sents / self.num_threads)
        threads = []
        if self.threading:
            tail = 0
            for i in range(self.num_threads):
                head = tail
                tail = tail + \
                    size if i != (self.num_threads-1) else num_sents-1
                thread = threading.Thread(target=self.tokenize_en_ja, args=[
                                          en_sents[head: tail], ja_sents[head: tail]])
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            self.tokenize_en_ja([en_sents[num_sents-1]],
                                [ja_sents[num_sents-1]])
        else:
            self.tokenize_en_ja(en_sents[:num_sents], ja_sents[:num_sents])

        bitexts = [sent for sents in self.out for sent in sents]
        en_ls, ja_ls = [], []
        for bitext in bitexts:
            en, ja = bitext.split('\t')
            en_ls.append(en.strip())
            ja_ls.append(ja.strip())

        return en_ls, ja_ls


if __name__ == "__main__":
    en_ls = ["I have to sleep.",
             "Michael is twenty years old today.",
             "The password is Muriel.",
             "I will come back soon."]
    ja_ls = ["私は眠らなければなりません。",
             "マイケルは今日２０歳になりました。",
             "パスワードは「Muiriel」です。",
             "まもなく私は戻って来ます。"
             ]

    t = Tokenization(threading=False)
    en_sents, ja_sents = t.tokenize(en_ls, ja_ls)
    print(en_sents)
    print(ja_sents)
