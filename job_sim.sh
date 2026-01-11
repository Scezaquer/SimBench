#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1

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
    --output_file results/verbalized_Qwen2.5-7B-Instruct_token_prob_pop.pkl \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --method token_prob
