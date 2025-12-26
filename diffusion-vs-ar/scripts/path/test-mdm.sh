exp=output/path/mdm-path
mkdir -p $exp

for dataset in path_test
do
topk_decoding=True
mkdir $exp/$dataset
CUDA_VISIBLE_DEVICES=0  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 75 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    > $exp/${dataset}/eval-TopK$topk_decoding.log
    
python scripts/path/cal_path_acc.py data/path_test.jsonl $exp/${dataset}/generated_predictions.jsonl > $exp/pd_acc.log
done
