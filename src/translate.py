import re
import unicodedata
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
import argparse


def preprocess_en(en, path_bpe_model, alpha=None):
    sp = spm.SentencePieceProcessor()
    sp.Load(path_bpe_model)

    x = unicodedata.normalize('NFKC', x)
    x = x.strip()
    x = ' '.join(x.split())
    if alpha is None:
        x = sp.encode(x, out_type=int)
    else:
        x = sp.encode(x, out_type=int, enable_sampling=True, alpha=alpha)
    x = ' '.join([sp.IdToPiece(i) for i in x])
    return x


def en2ja(en):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, defalut="../checkpoints/",
                        help="path to the directory which stores checkpoint files")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint_last.pt",
                        help="a checkpoint file to be loaded  e.g. checkpoint_best.pt  checkpoint_last.pt")
    parser.add_argument("--data_name_or_path", type=str, default="../data-bin",
                        help="a file's name or path where data are stored")
    parser.add_argument("--path_bpe_model", type=str, default="../bpe.model")
    args = parser.parse_args()

    model = TransformerModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path)

    en = preprocess_en(en, args.path_bpe_model)
    ja = model.translate(x, beam=5, lenpen=0.6)
    ja = ''.join(ja.split()).replace('▁', '').strip()

    return ja
