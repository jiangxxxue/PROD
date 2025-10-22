#!/bin/bash

# Running script for the copyrighted code unlearning task

lr=5e-6

Model="codellama/CodeLlama-7b-hf"
ModelName="CodeLlama-7b-hf"
ModelPath="learned_model"
DatasetPath="data/forget_set_100"
SaveModelPath="outputs/models/PROD_lr${lr}"

python PROD.py \
--model_name ${Model} \
--model_path ${ModelPath} \
--output_dir ${SaveModelPath} \
--train_data_path ${DatasetPath} \
--alpha 0.0 \
--num_train_epochs 10 \
--learning_rate ${lr} \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32 \
--logging_steps 1 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_train \
--save_strategy no || exit

OutputDir="outputs/results/PROD_lr${lr}"
suffix="2025"

for file in "$SaveModelPath"/*; do
filename=$(basename "$file")
echo "Filename: ${filename}, Path: ${file}"

python test_forget_quality.py \
--model_name ${ModelName} \
--model_path ${file} \
--dataset ${DatasetPath} \
--num-samples 1 \
--temperature 0.0 \
--output-dir ${OutputDir}/${filename}/forget_quality \
--output-file-suffix ${suffix}

python test_model_utility.py \
--model_name ${ModelName} \
--model_path ${file} \
--dataset "HumanEval" \
--num-samples 1 \
--temperature 0.0 \
--output-dir ${OutputDir}/${filename}/model_utility \
--output-file-suffix ${suffix}

python evaluate.py \
--dataset ${DatasetPath} \
--input_path ${OutputDir}/${filename}/model_utility/${DatasetPath}_${ModelName}_temp0.0_toppNone_topkNone_samples1_0shot_${suffix}.jsonl \
--truncate \
--eval_standard

done