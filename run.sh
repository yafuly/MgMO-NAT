export PYTHONPATH=""
RUN=""
MODEL=""
RESTORE=""
DATA=""
mkdir -p $MODEL

time=$(date +'%m:%d:%H:%M')
echo $RUN
# Training
gpu=0
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpu python $RUN/fairseq_cli/train.py  $DATA  --save-dir $MODEL  --tensorboard-logdir $MODEL/tb --eval-tokenized-bleu \
   --keep-interval-updates 5 --save-interval-updates 300 --batch-size-valid 10 --validate-interval-updates 300 --maximize-best-checkpoint-metric \
   --eval-bleu-remove-bpe --best-checkpoint-metric bleu --log-format simple --log-interval 100 \
   --eval-bleu --eval-bleu-detok space --keep-last-epochs 5 --keep-best-checkpoints 5  --fixed-validation-seed 7 --ddp-backend=no_c10d \
   --max-tokens 256  --update-freq 1 \
   --share-all-embeddings --decoder-learned-pos --encoder-learned-pos  --optimizer adam --adam-betas "(0.9,0.98)" --lr 2e-6 \
   --lr-scheduler fixed  \
   --apply-bert-init --weight-decay 0.01 \
   --clip-norm 2.0 --max-update 500000  --task translation_lev --criterion nat_mgmo_loss --arch cmlm_transformer --noise full_mask \
   --label-smoothing 0.1 \
   --activation-fn gelu --dropout 0.1  \
   --length-loss-factor 1 --pred-length-offset \
   --reset-optimizer \
   --reset-dataloader \
   --reset-meters \
   --restore-file $RESTORE \
   --n-sample 40 \
   --reward-alpha 0.005 \
   --max-length-bias 5 \
   --null-input \
   --rm-scale 8 \
   --len-loss \
   --reward-factor 1 2>&1 | tee $MODEL/train.log.$time


# Inference
beam=5 # Length beam: 5
test_model=checkpoint_best.pt
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$gpu python $RUN/fairseq_cli/generate.py \
    $DATA \
    --gen-subset test \
    --task translation_lev \
    --path $MODEL/$test_model  \
    --iter-decode-with-beam $beam \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 16 2>&1 | tee $MODEL/test1.$test_model.$B.log.$time
