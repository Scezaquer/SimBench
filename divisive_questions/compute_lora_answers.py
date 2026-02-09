
import os
import json
import torch
import glob
import argparse
from tqdm import tqdm
from unsloth import FastLanguageModel

def get_token_probabilities(model, tokenizer, prompt, target_tokens):
    """
    Calculates the probabilities of the target tokens given a prompt.
    Adapted from generate_answers.py
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = torch.nn.functional.softmax(last_token_logits, dim=-1)
        results = {}
        for token in target_tokens:
            # Handle tokenization of the option (e.g. "Yes" vs " Yes")
            # We enforce checking both with and without leading space if applicable, 
            # though usually the prompt formatting handles the space.
            # Here we follow the logic from generate_answers.py roughly
            
            # Simple encoding of the token
            encodings = tokenizer.encode(token, add_special_tokens=False)
            # Also try with a leading space if the tokenizer merges spaces
            underscored_encodings = tokenizer.encode(" " + token, add_special_tokens=False)
            
            probability_sum = 0
            # Sum probs of all tokens making up the word (roughly approximation if split, 
            # but usually single token for "Yes"/"No")
            # Actually generate_answers.py sums probability of specific encodings. 
            # If "Yes" is one token, it takes that prob.
            
            # We strictly check for the first token of the option
            if encodings:
                probability_sum += probabilities[encodings[0]].item()
            
            # The original script summed encodings and underscored encodings.
            # This handles "Yes" and " Yes" being different tokens.
            if underscored_encodings and underscored_encodings != encodings:
                 probability_sum += probabilities[underscored_encodings[0]].item()
                 
            results[token] = probability_sum
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate probability distributions for divisive questions using multiple LoRAs.")
    parser.add_argument("--base_model", type=str, default="marcelbinz/Llama-3.1-Minitaur-8B", help="Base model name or path.")
    parser.add_argument("--lora_dir", type=str, default="/home/mila/a/aurelien.buck-kaeffer/scratch/marcelbinz", help="Directory containing LoRA adapters.")
    parser.add_argument("--questions_file", type=str, default="divisive_questions/potential_questions.json", help="Path to the questions JSON file.")
    parser.add_argument("--output_file", type=str, default="results/divisive_questions_probabilities.json", help="Path to save the output JSON file.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit quantization.")
    args = parser.parse_args()

    print(f"Loading questions from {args.questions_file}")
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = args.load_in_4bit,
    )
    
    # Set up chat template if missing or for specific models
    if tokenizer.chat_template is None or "Qwen" in args.base_model:
        print("Using custom ChatML template.")
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    FastLanguageModel.for_inference(model)

    # Get list of LoRAs
    lora_paths = sorted([d for d in glob.glob(os.path.join(args.lora_dir, "*")) if os.path.isdir(d)])
    print(f"Found {len(lora_paths)} LoRAs in {args.lora_dir}")

    # Structure to hold results: list of question results
    # Each question result will contain the question text and a dictionary of distributions keyed by LoRA name
    all_results = []
    
    # Initialize result structure
    for q in questions:
        all_results.append({
            "question": q["question"],
            "distributions": {}
        })

    for lora_path in tqdm(lora_paths, desc="Processing LoRAs"):
        lora_name = os.path.basename(lora_path)
        # Sanitize adapter name for PEFT/Torch (replace dots with underscores)
        sanitized_adapter_name = lora_name.replace(".", "_")
        
        # Load LoRA adapter
        # Note: Unsloth models are PEFT models. 
        # We use standard PEFT load_adapter. 
        try:
            model.load_adapter(lora_path, adapter_name=sanitized_adapter_name)
            model.set_adapter(sanitized_adapter_name)
        except Exception as e:
            print(f"Error loading adapter {lora_name}: {e}")
            continue

        for idx, q in enumerate(questions):
            question_text = q["question"]
            options = q["options"]
            
            # Format Prompt
            # Using chat template as it's Llama 3.1
            messages = [{"role": "user", "content": question_text}]
            
            # add_generation_prompt=True ensures we get the header for the assistant
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Calculate probabilities
            raw_probs = get_token_probabilities(model, tokenizer, prompt, options)
            
            # Normalize
            total_prob = sum(raw_probs.values())
            if total_prob > 0:
                normalized_probs = {k: v / total_prob for k, v in raw_probs.items()}
            else:
                normalized_probs = {k: 0.0 for k in raw_probs} # Should not happen typically
            
            all_results[idx]["distributions"][lora_name] = normalized_probs
            
        # Unload adapter to free memory if needed, or just leave it for set_adapter to switch
        # model.delete_adapter(lora_name) 

    # Save results
    if not os.path.exists(os.path.dirname(args.output_file)) and os.path.dirname(args.output_file) != "":
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
