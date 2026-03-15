#!/bin/bash
#SBATCH --job-name=SimBench_LoRAs
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=main

nvidia-smi
lscpu

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
export UNSLOTH_CACHE_DIR=$SCRATCH/unsloth-cache
export LOCAL_WORKDIR=/home/mila/a/aurelien.buck-kaeffer/SimBench
export HF_HUB_OFFLINE=1
export HF_CACHE_LOCAL=$SCRATCH/HF-cache
LORA_DIR="/home/mila/a/aurelien.buck-kaeffer/scratch/Qwen"
BASE_MODEL="Qwen3.5-9B-Base"

echo "Using LoRA directory: $LORA_DIR"

for section in "$LORA_DIR"/Qwen3.5-9B-Base*; do
    if [ -d "$section" ]; then
        lora_name=$(basename "$section")
        echo "Processing LoRA: $lora_name"
        
        output_file="results/${lora_name}_token_prob_pop.pkl"
        
        if [ -f "$output_file" ]; then
             echo "Output file $output_file already exists. Skipping..."
             continue
        fi
        
        echo "Running generation for $lora_name -> $output_file"
        
        python generate_answers.py \
            --input_file SimBenchPop.pkl \
            --output_file "$output_file" \
            --model_name "$BASE_MODEL" \
            --method token_prob \
            --lora_path "$section"
            
    fi
done

echo "All LoRAs processed."
