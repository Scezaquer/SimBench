
import json
import pandas as pd
import argparse
import os
import numpy as np

def extract_lora_name(model_file):
    """
    Extracts the LoRA name from the Model_File string in the weights CSV.
    Example: token_prob_Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-15_token_prob_pop.pkl
    -> Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-15
    """
    name = model_file
    if name.startswith("token_prob_"):
        name = name[len("token_prob_"):]
    if name.endswith("_token_prob_pop.pkl"):
        name = name[:-len("_token_prob_pop.pkl")]
    return name

def calculate_entropy(probs):
    """Calculates the Shannon entropy of a probability distribution."""
    # Filter out zeros to avoid log(0)
    probs = [p for p in probs if p > 0]
    return -sum(p * np.log2(p) for p in probs)

def main():
    parser = argparse.ArgumentParser(description="Find the most divisive questions based on model distributions.")
    parser.add_argument("--input_file", type=str, default="divisive_questions_probabilities.json", help="Path to the distributions JSON.")
    parser.add_argument("--weights_file", type=str, help="Path to the weights CSV file (optional).")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Load weights
    weights = {}
    if args.weights_file:
        if os.path.exists(args.weights_file):
            df_weights = pd.read_csv(args.weights_file)
            for _, row in df_weights.iterrows():
                lora_name = extract_lora_name(row['Model_File'])
                weights[lora_name] = row['Weight']
            print(f"Loaded weights for {len(weights)} models from {args.weights_file}")
        else:
            print(f"Warning: Weights file {args.weights_file} not found. Using uniform weights.")

    results = []

    for item in data:
        question = item['question']
        distributions = item['distributions']
        
        # Determine available LoRAs that we have weights for (or all if uniform)
        available_loras = list(distributions.keys())
        
        # Initialize weighted answer probabilities
        # Get options from the first distribution
        options = list(distributions[available_loras[0]].keys())
        weighted_probs = {opt: 0.0 for opt in options}
        
        total_weight = 0.0
        
        for lora_name, dist in distributions.items():
            # If weights are provided, use them; otherwise use 1.0
            w = weights.get(lora_name, 1.0 if not weights else 0.0)
            
            if w == 0 and weights:
                continue
                
            for opt, prob in dist.items():
                weighted_probs[opt] += prob * w
            total_weight += w

        if total_weight == 0:
            print(f"Warning: No matching weights found for question: {question[:50]}...")
            continue

        # Normalize weighted probabilities
        final_probs = {opt: p / total_weight for opt, p in weighted_probs.items()}
        
        # Calculate metric: Entropy (higher is more divisive/uniform)
        # For binary, entropy is maximized at 0.5/0.5
        entropy = calculate_entropy(list(final_probs.values()))
        
        # Also calculate distance to 50-50 for binary specifically (as requested)
        # If not binary, we'll just stick to entropy as the sorting metric
        divisiveness_score = entropy 
        
        results.append({
            "question": question,
            "probs": final_probs,
            "entropy": entropy
        })

    # Sort by entropy descending (most divisive first)
    results.sort(key=lambda x: x['entropy'], reverse=True)

    print("\n" + "="*80)
    print(f"{'MOST DIVISIVE QUESTIONS':^80}")
    print("="*80 + "\n")

    for i, res in enumerate(results):
        print(f"{i+1}. {res['question']}")
        prob_str = ", ".join([f"{opt}: {p:.2%}" for opt, p in res['probs'].items()])
        print(f"   Distribution: {prob_str}")
        print(f"   Entropy: {res['entropy']:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
