import pandas as pd
import glob
import os
import re

results_dir = 'results'
pickle_files = glob.glob(os.path.join(results_dir, "*.pkl"))

for file_path in pickle_files:
    filename = os.path.basename(file_path)
    
    # Extract the descriptive part of the filename
    # Expected format: token_prob_{MODEL_OR_LORA}_token_prob_pop.pkl
    match = re.search(r'token_prob_(.+)_token_prob_pop\.pkl', filename)
    if match:
        new_model_name = match.group(1)
        print(f"Renaming model in {filename} to {new_model_name}")
        
        try:
            df = pd.read_pickle(file_path)
            df['Model'] = new_model_name
            df.to_pickle(file_path)
            print(f"  Successfully updated {file_path}")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    else:
        print(f"Skipping {filename}: does not match expected pattern")

print("Done.")
