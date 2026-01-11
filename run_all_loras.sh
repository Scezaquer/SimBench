#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=12:00:00
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

LORA_DIR="/home/s4yor1/scratch/qwen-loras"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Using LoRA directory: $LORA_DIR"

for section in "$LORA_DIR"/*; do
    if [ -d "$section" ]; then
        lora_name=$(basename "$section")
        echo "Processing LoRA: $lora_name"
        
        output_file="results/token_prob_${lora_name}_token_prob_pop.pkl"
        
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
