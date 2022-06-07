import re
from datasets import load_dataset
import os
from tqdm import tqdm
import json

def dl_tatoeba(repo_path):
    ds_path = os.path.join(repo_path, "/corpus/data")
    ds_dict = load_dataset("tatoeba", lang1="en", lang2="ja")
    ds = ds_dict["train"]
    ds.info.write_to_directory(ds_path)
    ds.to_json(os.path.join(ds_path, "/en-ja.json"))

def reform_json(file1, file2):
    with open(file1, "r", encoding="utf-8") as fr, open(file2, "w", encoding="utf-8") as fw:
        num_sents = int(os.popen("wc -l %s" % file1).read().strip().split()[0])
        fw.write('{\n  "bitexts-en-ja": [\n')
        for i in tqdm(range(0, num_sents)):
           line = fr.readline()
           if i != num_sents-1:
               fw.write("  " + line[:-1] + ",\n")
           else:
               fw.write("  " + line[:-1] + "\n")
        fw.write(' ]\n}')

def json2list(repo_path):
    ds_path = os.path.join(repo_path, "/corpus/data/en-ja.json")
    tmp_path = os.path.join(repo_path, "/corpus/data/dataset.json")

    # jsonファイルをreformatした後、保存し直す
    reform_json(ds_path, tmp_path)
    with open(tmp_path, 'r', encoding='utf-8') as fr, open(ds_path, "w", encoding="utf-8") as fw:
        data = json.load(fr)
        json.dump(data, fw, indent=2, ensure_ascii=False)

if __name__=="__main__":
    dl_tatoeba("/home/hiroshi/Machine_Translation_Proto")
    json2list("/home/hiroshi/Machine_Translation_Proto")
