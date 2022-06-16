import re
import sacremoses as sm
import unicodedata
import MeCab
from typing import List
import multiprocessing as mp
import os


class Tokenization():
    def __init__(self, multiproc=False, num_procs=4):
        self.multiproc = mp.Value('i', int(multiproc))
        self.num_procs = mp.Value('i', num_procs)

    def tokenize_en(self, en_sents: List[str]):
        mt = sm.MosesTokenizer(lang='en')
        en_ls = []
        for en in en_sents:
            en = unicodedata.normalize("NFKC", en)
            en = re.sub(
                mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
            en = mt.tokenize(en, escape=False)
            en = ' '.join(en).lower()
            en_ls.append(en)
        return en_ls

    def tokenize_ja(self, ja_sents: List[str]):
        mecab = MeCab.Tagger("-Owakati")
        ja_ls = []
        for ja in ja_sents:
            ja = unicodedata.normalize("NFKC", ja)
            ja = mecab.parse(ja)
            ja_ls.append(ja)
        return ja_ls

    def tokenize_en_ja(self, queue, en_sents: List[str], ja_sents: List[str]):
        en_ls = self.tokenize_en(en_sents)
        ja_ls = self.tokenize_ja(ja_sents)
        queue.put([en.replace('\t', '') + '\t' + ja.replace('\t', '')
                   for en, ja in zip(en_ls, ja_ls)])
        print("Tokenization [Process ID: {}] has finished.".format(os.getpid()))

    def tokenize(self, en_sents: List[str], ja_sents: List[str]):
        queue = mp.Queue()
        num_sents = min(len(en_sents), len(ja_sents))
        size = int(num_sents / self.num_procs.value)
        if self.multiproc.value:
            tail = 0
            for i in range(self.num_procs.value):
                head = size * i
                tail = size * (i+1) if i != (self.num_procs.value-1) else num_sents
                proc = mp.Process(target=self.tokenize_en_ja, args=[queue,
                                                                    en_sents[head: tail], ja_sents[head: tail]])
                proc.start()

        else:
            self.num_procs.value = 1
            self.tokenize_en_ja(
                queue, en_sents[:num_sents], ja_sents[:num_sents])

        out = []
        for i in range(self.num_procs.value):
            out.append(queue.get())
        bitexts = [sent for sents in out for sent in sents]
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

    t = Tokenization(multiproc=True, num_procs=2)
    en_sents, ja_sents = t.tokenize(en_ls, ja_ls)
    print(en_sents)
    print(ja_sents)
