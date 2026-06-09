#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --job-name=mirage-shuf
#SBATCH --output=/home/scur0259/mirage/logs/ablate_shuffle_%j.out
#SBATCH --error=/home/scur0259/mirage/logs/ablate_shuffle_%j.err

module load 2023
module unload PyTorch
cd /home/scur0259/mirage && source venv/bin/activate
export HF_HOME=/scratch-shared/scur0259/hf_cache
mkdir -p logs

echo "============================================================"
echo "RQ3 Ablation: shuffle — 1000 samples"
echo "============================================================"

rm -rf /scratch-shared/scur0259/mirage_ablation/shuffle/
python3 ablate.py --type shuffle --num-samples 1000

echo "Done: $(date)"
