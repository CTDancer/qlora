export WANDB_API_KEY=b7b6ecceb6854bd12f58809f18264f979509d13b
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0,4
python distill.py \
    --model_name_or_path baichuan-inc/Baichuan-7B \
    --use_auth \
    --output_dir ./output/baichuan-alpaca-cleabn-7 \
    --logging_steps 100 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 10 \
    --max_eval_samples 10 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 1 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing=True \
    --ddp_find_unused_parameters=False \
    --dataset /home/dqwang/scratch/tongchen/qlora/distilled_dataset.json \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 5 \
    --eval_steps 3 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --save_dir /shared/dqwang/scratch/tongchen/qlora/distill \
    --lr_teacher=1e-2 \
    --lr_text=1e-1 \
    --lr_label=1e-1 \
    --lr_lr=1e-3 \
    --syn_steps=1 \
    --expert_dir '/shared/dqwang/scratch/tongchen/qlora/baichuan' \
    --batch_syn=8 \
    --trust_remote_code True