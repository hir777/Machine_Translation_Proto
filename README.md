# Machine_Translation_Proto

## データの前処理を行う
```
bash pre-process.sh
```

## トークン使用率 58  ## 残り会話数(概算) 7

## トークン使用率 {int(remaining_tokens/max_tokens * 100)}％    ## 残り会話数(概算) {remaining_talks}

## モデルにデータセットを学習させる

```bash
fairseq-train \
    data-bin \
    --fp16 \
    --save-interval 5 \
    --log-interval 1 \
    --log-format simple \
    --max-epoch 30 \
    --update-freq 1 \
    --max-tokens 4000 \
    --arch transformer \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 8 \
    --encoder-layers 8 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 2048 \
    --decoder-attention-heads 8 \
    --decoder-layers 8 \
    --share-all-embeddings \
    --dropout 0.3 \
    --attention-dropout 0.0 \
    --activation-dropout 0.0 \
    --activation-fn relu \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --lr 0.0015 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --clip-norm 1.0 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.3 \
    | tee train.log
```

## 学習済みモデルを利用する

```bash
fairseq-interactive data-bin \
    --buffer-size 1024 \
    --batch-size 128 \
    --path checkpoints/checkpoint30.pt \
    --beam 5 \
    --lenpen 0.6 \
    < test.en \
    | grep '^H' \
    | cut -f 3 \
    | python src/decode.py \
    | tee output.txt \
    | sacrebleu corpus/test.ja
```
