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
source ../ENV/bin/activate
export HF_CACHE_LOCAL=/home/mila/a/aurelien.buck-kaeffer/scratch/HF-cache
export LOCAL_WORKDIR=/home/mila/a/aurelien.buck-kaeffer/SimBench_release
export HF_HUB_OFFLINE=1

LORA_DIR="/home/mila/a/aurelien.buck-kaeffer/scratch/Minitaur"
BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"

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
