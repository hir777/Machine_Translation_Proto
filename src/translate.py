import re
import unicodedata
import sentencepiece as spm
from fairseq.models.transformer import TransformerModel
import argparse
from sacremoses import MosesTokenizer


def preprocess_en(en, sp, mt):
    en = unicodedata.normalize('NFKC', en)
    en = re.sub(mt.AGGRESSIVE_HYPHEN_SPLIT[0], r'\1 - ', en)
    en = mt.tokenize(en, escape=False)
    en = en.strip().lower()
    en = ' '.join(sp.encode(en, out_type="str"))
    return en


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, defalut="../checkpoints/",
                        help="path to the directory which stores checkpoint files")
    parser.add_argument("--checkpoint_file", type=str, default="checkpoint_last.pt",
                        help="a checkpoint file to be loaded  e.g. checkpoint_best.pt  checkpoint_last.pt")
    parser.add_argument("--data_name_or_path", type=str, default="../data-bin",
                        help="a file's name or path where data are stored")
    parser.add_argument("--path_bpe_model", type=str, default="../bpe.model")
    parser.add_argument("en", type=str, default="Hello World.")
    args = parser.parse_args()

    model = TransformerModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path)

    mt = MosesTokenizer(lang="en")
    sp = spm.SentencePieceProcessor(model_file=args.path_bpe_model)

    en = preprocess_en(args.en, sp, mt)
    ja = model.translate(args.en, beam=5, lenpen=0.6)
    ja = ''.join(ja.split()).replace('▁', '').strip()

    print(ja)
