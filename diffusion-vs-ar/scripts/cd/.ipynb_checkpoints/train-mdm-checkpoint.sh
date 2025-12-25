export WANDB_DISABLED=true

exp=output/cd3/mdm-alpha0.25-gamma2-bs1024-lr3e-4-ep600-T20-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

CUDA_VISIBLE_DEVICES=0 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --main_process_port 20098 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset cd3_train \
    --finetuning_type full \
    --cutoff_len 64 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 2048 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 256 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 3e-4 \
    --num_train_epochs 600.0 \
    --plot_loss \
    --run_name ${dataset}_prefix \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 2 \
    > $exp/train.log

for dataset in cd3_test
do
topk_decoding=True
mkdir $exp/$dataset
CUDA_VISIBLE_DEVICES=0  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 64 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $exp/${dataset}/eval-TopK$topk_decoding.log
done
