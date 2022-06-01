import os

dir_data = "~/Machine_Translation_Proto/content/"
if not os.path.exists(dir_data):
  os.mkdir(dir_data)
os.chdir(dir_data)
os.system('wget　https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/pretrained_models/en-ja/small.tar.gz')
os.system('tar -xf small.tar.gz')
os.system('rm small.tar.gz')
