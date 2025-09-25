export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --lga \
  --eps 1e-2 \
  --d_G 64 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --pct_start 0.4 \
  --learning_rate 1e-4 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --lga \
  --eps 1e-2 \
  --d_G 64 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --pct_start 0.4 \
  --learning_rate 1e-4 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --lga \
  --eps 1e-2 \
  --d_G 64 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --pct_start 0.4 \
  --learning_rate 1e-4 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 \
  --lga \
  --eps 1e-2 \
  --d_G 64 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --pct_start 0.4 \
  --learning_rate 1e-4 \
