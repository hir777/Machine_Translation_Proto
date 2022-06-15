from tqdm import tqdm
import os

def dl_WikiMatrix(repo_path):
    num_sents = 2156862
    os.system("cd {}".format(os.path.join(repo_path, "corpus/data/")))
    os.system("wget --progress=bar https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-ja.tsv.gz")
    os.system("gunzip WikiMatrix.en-ja.tsv.gz")
    os.system("rm WikiMatrix.en-ja.tsv.gz")

    en_ls, ja_ls = [], []
    with open("WikiMatrix.en-ja.tsv", 'r') as f:
        print("\nDownloading WikiMatrix dataset...")
        for line in tqdm(f, total=num_sents):
            _, en, ja = f.readline().rstrip().split('\t')
            en_ls.append(en + '\n')
            ja_ls.append(ja + '\n')
    
    return en_ls, ja_ls

if __name__=="__main__":
    en_ls, ja_ls = dl_WikiMatrix("/home/hiroshi/Machine_Translation_Proto/corpus/data/")
    for en, ja in zip(en_ls[1000:1010], ja_ls[1000:1010]):
        print("en: %s" % en)
        print("ja: %s \n" % ja)
