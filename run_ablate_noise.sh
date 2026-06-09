#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --job-name=mirage-noise
#SBATCH --output=/home/scur0259/mirage/logs/ablate_noise_%j.out
#SBATCH --error=/home/scur0259/mirage/logs/ablate_noise_%j.err

module load 2023
module unload PyTorch
cd /home/scur0259/mirage && source venv/bin/activate
export HF_HOME=/scratch-shared/scur0259/hf_cache
mkdir -p logs

echo "============================================================"
echo "RQ3 Ablation: noise (σ=0.1) — 1000 samples"
echo "============================================================"

rm -rf /scratch-shared/scur0259/mirage_ablation/noise/
python3 ablate.py --type noise --num-samples 1000 --noise-std 0.1

echo "Done: $(date)"
