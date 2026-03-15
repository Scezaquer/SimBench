#!/bin/bash
#SBATCH --job-name=SimBench
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=unkillable

nvidia-smi
lscpu

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
export UNSLOTH_CACHE_DIR=$SCRATCH/unsloth-cache
export LOCAL_WORKDIR=/home/mila/a/aurelien.buck-kaeffer/SimBench
export HF_CACHE_LOCAL=$SCRATCH/HF-cache
export HF_HUB_OFFLINE=1
echo "Starting script"

python generate_answers.py \
    --input_file SimBenchPop.pkl \
    --output_file results/Qwen3-8B_token_prob_pop.pkl \
    --model_name Qwen/Qwen3-8B \
    --method token_prob
