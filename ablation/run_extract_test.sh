#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --time=00:45:00
#SBATCH --gpus=1
#SBATCH --job-name=mirage-test-heat
#SBATCH --output=/home/scur0259/mirage/logs/extract_test_%j.out
#SBATCH --error=/home/scur0259/mirage/logs/extract_test_%j.err

# Test-set heatmap extraction (eager + dummy pixel_values_latent). One clean forward
# per sample -> ~5 min for 400. Resumable by sample_id, so DELETE the old
# attention-less output dir first or it will skip everything.

module load 2023
module unload PyTorch
cd /home/scur0259/mirage && source venv/bin/activate
export HF_HOME=/scratch-shared/scur0259/hf_cache
mkdir -p logs

echo "TEST heatmap extraction (eager, dummy pixel_values_latent)"
python3 ablation/extract_test.py \
    --num-samples 400 \
    --source-jsonl /home/scur0259/mirage/data/vsp_spatial_planning/test_direct_with_oi.jsonl \
    --output-dir /scratch-shared/scur0259/mirage_test_extracted

echo "Done: $(date)"
