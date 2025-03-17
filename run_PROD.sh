bs=32
lr=5e-6

Model="codellama/CodeLlama-7b-hf"
ModelName="CodeLlama-7b-hf"
DatasetPath="data/forget_data"
SaveModelPath="outputs/models"
OutputDir="outputs/results"
suffix="0315"

python PROD.py \
--model_name ${Model} \
--model_path ${ModelPath} \
--output_dir ${SaveModelPath} \
--train_data_path ${DatasetPath} \
--top_p 0.8 \
--alpha 0.0 \
--N 1 \
--num_train_epochs 10 \
--learning_rate ${lr} \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps ${bs} \
--logging_steps 1 \
--save_total_limit 2 \
--overwrite_output_dir \
--do_train \
--save_strategy no



for file in "$SaveModelPath"/*; do
filename=$(basename "$file")
echo "Filename: ${filename}, Path: ${file}"

python test_forget_quality.py \
--model_name ${Model} \
--model_path ${file} \
--dataset ${DatasetPath} \
--num-samples 1 \
--temperature 0.0 \
--output-dir ${OutputDir}/${filename}/forget_quality \
--output-file-suffix ${suffix} \
--ppl_only

python test_model_utility.py \
--model_name ${Model} \
--model_path ${file} \
--dataset "HumanEval" \
--num-samples 1 \
--temperature 0.0 \
--output-dir ${OutputDir}/${filename}/model_utility \
--output-file-suffix ${suffix}

python evaluate.py \
--dataset "HumanEval" \
--input_path ${OutputDir}/${filename}/model_utility/HumanEval_${ModelName}_temp0.0_toppNone_topkNone_samples1_0shot_${suffix}.jsonl \
--truncate \
--eval_standard

done