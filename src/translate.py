import re
import unicodedata
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
import argparse
from sacremoses import MosesTokenizer


def preprocess_en(en:str, sp, mt):
    en = unicodedata.normalize('NFKC', en)
    en = re.sub(mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
    en = mt.tokenize(en, escape=False)
    en = ' '.join(en).lower()
    en = ' '.join(sp.encode(en, out_type="str"))
    return en


def en2ja(checkpoint_dir, checkpoint_file, data_name_or_path, path_bpe_model, en):
    model = TransformerModel.from_pretrained(
        checkpoint_dir,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_name_or_path)

    mt = MosesTokenizer(lang="en")
    sp = spm.SentencePieceProcessor(model_file=path_bpe_model)

    en = preprocess_en(en, sp, mt)
    ja = model.translate(en, beam=5, lenpen=0.6)
    ja = ''.join(ja.split()).replace('▁', '').strip()

    return ja
