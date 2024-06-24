@echo off
set model_name=LLM4MRSI
set env_name=llm4sst

REM Call conda activate to ensure the environment is activated correctly
call conda activate %env_name%

for %%b in (16) do (
    for %%l in (0.001) do (
        for %%g in (1) do (
            for %%e in (576) do (
                python -u run_win.py ^
                    --task_name imputation ^
                    --is_training 1 ^
                    --root_path ./dataset/ ^
                    --data_path cha_all1_down1.csv ^
                    --model_id all_all1_down1_mask_0.1 ^
                    --mask_rate 0.1 ^
                    --model %model_name% ^
                    --data multisource ^
                    --features M ^
                    --seq_len 24 ^
                    --label_len 0 ^
                    --pred_len 0 ^
                    --patch_size 1 ^
                    --d_ff 768 ^
                    --stride 1 ^
                    --gpt_layer %%g ^
                    --d_model 768 ^
                    --enc_in %%e ^
                    --dec_in %%e ^
                    --c_out 576 ^
                    --batch_size %%b ^
                    --train_epoch 100 ^
                    --des 'Exp' ^
                    --itr 5 ^
                    --learning_rate %%l
            )
        )
    )
)

REM Deactivate the environment after the script runs
call conda deactivate
