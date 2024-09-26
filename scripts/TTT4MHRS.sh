model_name=TTT4MHRS
#32 64 128 256
for batch_size in 16 #32 64 128 256
do
for lr in 0.001
do
for gpt_layer in 3
do
for enc_in in 576
do
for source in cha,par,sst
do
for last_fusion in TSConv2d  #WithoutFusion V_DAB ChannelAttention GroupConv STAR ShuffleConv TSConv2d TSDeformConv2d
do
for ttt_style in TTTLinear TTTMLP
do
for is_invert in 0 1
do

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --log_dir ./logs/ \
  --data_path all41.csv \
  --log_name ttt_scale.txt \
  --source_names $source \
  --model_id ${last_fusion}_${ttt_style}_is_invert_${is_invert}_down1_mask_0.2 \
  --mask_rate 0.2 \
  --model $model_name \
  --ttt_style ${ttt_style} \
  --is_invert $is_invert \
  --last_fusion $last_fusion \
  --data multisource \
  --features M \
  --seq_len 90 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --d_ff 768 \
  --len_dff 256 \
  --stride 1 \
  --gpt_layer $gpt_layer \
  --d_model 768 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 576 \
  --batch_size $batch_size \
  --train_epoch 100 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate $lr \
  --devices 8 \
  --use_multi_gpu

done
done
done
done
done
done
done
done