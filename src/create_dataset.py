import os


def write_files(bitexts, file_en, file_ja):
    with open(file_en) as f_en, open(file_ja) as f_ja:
        for bitext in bitexts:
            f_en.write(bitext[0])
            f_ja.write(bitexts[1])


def read_file(data_path):
    bitexts = []
    with open(data_path) as f:
        line = f.readline().split(' ')
        bitexts.append([line[3], line[4]])
    
    return bitexts


if __name__ == '__main__':

    row_data = "~/content/en-ja.bicleaner05.txt"
    dest = "~/Machine_Translation_Proto/content/"

    bitexts = read_file(row_data)
    write_files(bitexts[0:700000], os.path.join(dest, "train.en"), os.path.join(dest, "train.ja"))
    write_files(bitexts[700000:900000], os.path.join(dest, "dev.en"), os.path.join(dest, "dev.ja"))
    write_files(bitexts[900000:1000000], os.path.join(dest, "dev.en"), os.path.join(dest, "train.ja"))
