#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --time=03:30:00
#SBATCH --gpus=1
#SBATCH --job-name=regen-gen
#SBATCH --array=0-3
#SBATCH --output=/home/scur0259/mirage/logs/regen_gen_%A_%a.out
#SBATCH --error=/home/scur0259/mirage/logs/regen_gen_%A_%a.err

# Free-running (greedy) ablation generation, batch_size=1 (model's latent
# self-compute is not batch-safe). 4-way sample shard via the array index;
# all tasks write the SAME resumable output dir (per-sample files, disjoint
# ranges -> no collisions). KL-pruning does the heavy lifting.
#
# Submit:
#   sbatch ablation/run_regen_gen.sh test     # 400 mazes
#   sbatch ablation/run_regen_gen.sh train    # 996 mazes

module load 2023
module unload PyTorch
cd /home/scur0259/mirage && source venv/bin/activate
export HF_HOME=/scratch-shared/scur0259/hf_cache
mkdir -p logs

SPLIT=${1:-test}
if [ "$SPLIT" = "train" ]; then
    N=996
    SRC=/home/scur0259/mirage/data/vsp_spatial_planning/train_direct_with_oi.jsonl
    KL=/home/scur0259/ablated_plans_dist.jsonl
    OUT=/scratch-shared/scur0259/mirage_train_plans_gen
else
    N=400
    SRC=/home/scur0259/mirage/data/vsp_spatial_planning/test_direct_with_oi.jsonl
    KL=/home/scur0259/test_plans_dist.jsonl
    OUT=/scratch-shared/scur0259/mirage_test_plans_gen
fi

NSHARDS=4
SHARD=$(( (N + NSHARDS - 1) / NSHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * SHARD ))

echo "============================================================"
echo "regen_gen | split=$SPLIT | shard $SLURM_ARRAY_TASK_ID/$((NSHARDS-1))"
echo "  samples $START .. $((START + SHARD - 1))  (N=$N, shard=$SHARD)"
echo "  out=$OUT  src=$SRC  kl=$KL"
echo "============================================================"

python3 ablation/regen_gen.py \
    --source-jsonl "$SRC" \
    --kl-source "$KL" \
    --output-dir "$OUT" \
    --start "$START" --num-samples "$SHARD" \
    --batch-size 1 --kl-threshold 1e-3 --max-new-tokens 64

echo "Done: $(date)"
