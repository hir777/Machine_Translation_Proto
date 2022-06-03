import os
import re
import argparse

def find_ja(str):
    pattern = r"[\u3041-\u309F\u30A1-\u30FF\uFF66-\uFF9F\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF\u3000-\u303F\s]+"
    ja_sents = re.findall(pattern, str)
    ja_sents = [re.sub(r"\s", "", sent) for sent in ja_sents]
    
    return "".join(ja_sents) + "\n"

def find_en(str):
    pattern = r"[a-zA-Z,.!?[\]()\-\s]+"
    en_sents = re.findall(pattern, str)

    return "".join(en_sents)

def write_files(bitexts, file_en, file_ja):
    with open(file_en, 'a') as f_en, open(file_ja, 'a') as f_ja:
        for bitext in bitexts:
            f_en.write(bitext[0])
            f_ja.write(bitext[1])

def read_file(data_path, size):
    bitexts = []
    with open(data_path, 'r') as f:
        for i in range(0, size):
            line = f.readline()
            bitexts.append([find_en(line), find_ja(line)])
    
    return bitexts

if __name__ == '__main__':
    row_data = "/content/split/"
    dest = "/Machine_Translation_Proto/corpus/"

    bitexts = read_file(os.path.join(row_data, "train"), 100)
    write_files(bitexts, os.path.join(dest, "train.en"), os.path.join(dest, "train.ja"))

    bitexts = read_file(os.path.join(row_data, "dev"), 100)
    write_files(bitexts, os.path.join(dest, "dev.en"), os.path.join(dest, "dev.ja"))

    bitexts = read_file(os.path.join(row_data, "test"), 100)
    write_files(bitexts, os.path.join(dest, "test.en"), os.path.join(dest, "test.ja"))
