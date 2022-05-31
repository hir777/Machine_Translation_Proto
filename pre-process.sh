set -ex

# 学習用・検証用・テスト用データのPATHを指定
TRAIN_EN=/Machine_Translation_Proto/corpus/train.en
TRAIN_JA=/Machine_Translation_Proto/corpus/train.ja
VALID_EN=/Machine_Translation_Proto/corpus/valid.en
VALID_JA=/Machine_Translation_Proto/corpus/valid.ja
TEST_EN=/Machine_Translation_Proto/corpus/test.en

# 学習用データセットを用いてSentencePieceを学習させる
cat $TRAIN_EN $TRAIN_JA > train.enja
python src/sp_learn.py --input train.enja --prefix bpe --vocab-size 4000 --character-coverage 0.9995 --threads 1

# 学習済みのSentencePieceを用いて各データセットをエンコードする
encode () {
    python src/encode.py --model bpe.model
}

encode < $TRAIN_EN > train.en
encode < $TRAIN_JA > train.ja
encode < $VALID_EN > valid.en
encode < $VALID_JA > valid.ja
encode < $TEST_EN > test.en

# fairseqの前処理用コマンドを実行する
fairseq-preprocess -s en -t ja \
    --trainpref train \
    --validpref valid \
    --destdir data-bin \
    --joined-dictionary
