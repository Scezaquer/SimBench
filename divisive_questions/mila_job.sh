#!/bin/bash
#SBATCH --job-name=DivisiveQuestions
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=unkillable

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
export UNSLOTH_CACHE_DIR=$SCRATCH/unsloth-cache
export LOCAL_WORKDIR=/home/mila/a/aurelien.buck-kaeffer/SimBench
# export HF_HUB_OFFLINE=1
export HF_CACHE_LOCAL=$SCRATCH/HF-cache

python compute_lora_answers.py \
    --base_model "marcelbinz/Llama-3.1-Minitaur-8B" \
    --lora_dir "/home/mila/a/aurelien.buck-kaeffer/scratch/marcelbinz" \
    --questions_file "potential_questions.json" \
    --output_file "divisive_questions_probabilities.json"