# Training stage 1

python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage1 \
    --data_path ./data/sample.jsonl \
    --log_file ./log.txt \
    --save_model_path ./checkpoints/model_stage1 

# Training stage 2

python src/main.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --epochs 10 \
    --task vsp-spatial-reasoning \
    --latent_size 4 \
    --stage stage2 \
    --data_path ./data/sample.jsonl \
    --log_file ./log.txt \
    --load_model_path ./checkpoints/model_stage1 \
    --save_model_path ./checkpoints/model_stage2