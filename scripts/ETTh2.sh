export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --d_model 16 \
  --d_ff 128\
  --n_heads 4 \
  --fc_dropout 0.3 \
  --dropout 0.3 \
  --des 'Exp' \
  --lga \
  --itr 1 \
  --eps 1e-2 \
  --d_G 32 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --schedulefree \
  --learning_rate 1e-3 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --d_model 16 \
  --d_ff 128\
  --n_heads 4 \
  --fc_dropout 0.3 \
  --dropout 0.3 \
  --des 'Exp' \
  --lga \
  --itr 1 \
  --eps 1e-2 \
  --d_G 32 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --schedulefree \
  --learning_rate 1e-3 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --d_model 16 \
  --d_ff 128\
  --n_heads 4 \
  --fc_dropout 0.3 \
  --dropout 0.3 \
  --des 'Exp' \
  --lga \
  --itr 1 \
  --eps 1e-2 \
  --d_G 32 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --schedulefree \
  --learning_rate 1e-3 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_336_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 7 \
  --d_model 16 \
  --d_ff 128\
  --n_heads 4 \
  --fc_dropout 0.3 \
  --dropout 0.3 \
  --des 'Exp' \
  --lga \
  --itr 1 \
  --eps 1e-2 \
  --d_G 32 \
  --batch_size 128 \
  --train_epochs 100 \
  --patience 10 \
  --schedulefree \
  --learning_rate 1e-3 \