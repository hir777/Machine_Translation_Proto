set -ex

TRAIN_EN=/Machine_Translation_Proto/corpus/train.en
TRAIN_JA=/Machine_Translation_Proto/corpus/train.ja
VALID_EN=/Machine_Translation_Proto/corpus/valid.en
VALID_JA=/Machine_Translation_Proto/corpus/valid.ja
TEST_EN=/Machine_Translation_Proto/corpus/test.en

cat $TRAIN_EN $TRAIN_JA > train.enja
python src/sp_learn.py --input train.enja --prefix bpe --vocab-size 4000 --character-coverage 0.9995 --threads 1

encode () {
    python src/encode.py --model bpe.model
}

encode < $TRAIN_EN > train.en
encode < $TRAIN_JA > train.ja
encode < $VALID_EN > valid.en
encode < $VALID_JA > valid.ja
encode < $TEST_EN > test.en

fairseq-preprocess -s en -t ja \
    --trainpref train \
    --validpref valid \
    --destdir data-bin \
    --joined-dictionary
