export WANDB_DISABLED=true

exp=output/cd4/llama-13b-bs1024-lr3e-4-ep40-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 \
src/train_bash.py \
    --stage sft --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path /cache/llama-13b \
    --do_train \
    --save_safetensors False \
    --dataset cd4_train \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 16 \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 32 \
    --per_gpu_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --num_train_epochs 40.0 \
    --plot_loss \
    --fp16 \
    --preprocessing_num_workers 8 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    > $exp/train.log

for dataset in cd4_tot24 cd4_test
do
mkdir $exp/${dataset}
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 \
src/train_bash.py \
    --stage sft --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path /cache/llama-13b \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora_rank 16 \
    --do_predict \
    --save_safetensors False \
    --cutoff_len 64 \
    --per_device_eval_batch_size 4 \
    --dataset $dataset \
    --finetuning_type lora \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --predict_with_generate \
    --adapter_name_or_path $exp \
    --max_new_tokens 32 \
> $exp/${dataset}/eval.log
done