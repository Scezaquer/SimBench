#!/bin/bash
#SBATCH --job-name=SimBench
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=unkillable

nvidia-smi
lscpu

module load python/3.11
module load scipy-stack

echo "Activating virtual environment..."
source ../concordia/ENV-concordia/bin/activate
export HF_CACHE_LOCAL=/home/s4yor1/scratch/HF-cache
export LOCAL_WORKDIR=/home/s4yor1/SimBench_release
export HF_HUB_OFFLINE=1
echo "Starting script"

python generate_answers.py \
    --input_file SimBenchPop.pkl \
    --output_file results/token_prob_marcelbinz-Llama-3.1-Minitaur-8B_token_prob_pop.pkl \
    --model_name marcelbinz/Llama-3.1-Minitaur-8B \
    --method token_prob
