import os
import glob
import re
import pandas as pd
import numpy as np
import cvxpy as cp

def parse_distribution(dist_entry, all_keys):
    """
    Convert a dictionary entry (human or model) into a fixed-size probability vector.
    dist_entry: dict, e.g., {'A': 10, 'B': 20}
    all_keys: list of keys to ensure consistent ordering, e.g., ['A', 'B', 'C', 'D', 'E']
    """
    if not isinstance(dist_entry, dict):
        return np.zeros(len(all_keys))
    
    # Extract values in order
    values = [float(dist_entry.get(k, 0.0)) for k in all_keys]
    values = np.array(values)
    
    # Normalize
    total = values.sum()
    if total > 0:
        return values / total
    return values

def make_hard_distribution(prob_vector):
    """
    Convert a probability vector into a one-hot vector where the max element is 1.
    If multiple entries have the max value, np.argmax selects the first one.
    """
    if np.sum(prob_vector) == 0:
        return prob_vector
    
    max_idx = np.argmax(prob_vector)
    hard_vector = np.zeros_like(prob_vector)
    hard_vector[max_idx] = 1.0
    return hard_vector

def load_data(results_dir):
    # Find all relevant files
    pattern = os.path.join(results_dir, "token_prob_Qwen2.5-7B-Instruct-lora-finetuned-*-no-focal_token_prob_pop.pkl")
    files = glob.glob(pattern)
    files.sort(key=lambda x: int(re.search(r'finetuned-(\d+)-', x).group(1)) if re.search(r'finetuned-(\d+)-', x) else -1)
    
    if not files:
        print("No files found matching the pattern.")
        return None, None, None

    print(f"Found {len(files)} model files.")

    # Load the first file to establish the target M and row ordering
    print(f"Loading reference (and first model) from: {files[0]}")
    df_ref = pd.read_pickle(files[0])
    
    possible_keys = set()
    for entry in df_ref['human_answer']:
        if isinstance(entry, dict):
            possible_keys.update(entry.keys())
    
    all_keys = sorted(list(possible_keys))
    print(f"Detected option keys: {all_keys}")

    # Build Target Matrix M (Hardened)
    M_list = []
    
    # We also keep the original for SimBench score calculation? 
    # Or should we calculate SimBench using the Hard target? 
    # The user request is "minimize the frobenius norm with this new matrix". 
    # Usually SimBench metric is defined against the *actual* human distribution.
    # But for optimization, we use the hard target.
    # I'll store both M_hard (for optimization) and M_soft (for reference/legacy simbench).
    M_soft_list = []
    
    for _, row in df_ref.iterrows():
        soft_dist = parse_distribution(row['human_answer'], all_keys)
        M_soft_list.append(soft_dist)
        
        # Hard construction
        M_list.append(make_hard_distribution(soft_dist))
        
    M = np.array(M_list)
    M_soft = np.array(M_soft_list)
    
    print(f"Target Matrix M (Hard) shape: {M.shape}")

    # Extract dataset names and compute Uniform TV for baseline normalization
    dataset_names = df_ref['dataset_name'].values
    
    # Pre-calculate TV_Uniform for each row (needed for SimBench score)
    # This part relies on M_soft logic usually, as baseline.
    tv_uniform_list = []
    dataset_norms = {}
    
    for i, row in df_ref.iterrows():
        # Baseline TV uses actual human answer, not hard answer?
        # SimBench standard matches human dist.
        # But if we optimized for Hard, maybe the user considers "max human answer" the target.
        # I will calculate standard SimBench norms using M_soft to be consistent with previous script scores.
        h_vals = list(row['human_answer'].values()) if isinstance(row['human_answer'], dict) else []
        n_opts = len(h_vals)
        
        if n_opts > 1:
            u_dist = np.ones(n_opts) / n_opts
            h_dist_local = np.array(h_vals, dtype=float)
            if h_dist_local.sum() > 0:
                h_dist_local /= h_dist_local.sum()
            else:
                pass 
                
            tv_uni = 0.5 * np.sum(np.abs(h_dist_local - u_dist))
        else:
            tv_uni = np.nan
        
        tv_uniform_list.append(tv_uni)

    # Compute Dataset Norms (Average TV_Uniform per dataset)
    df_meta = pd.DataFrame({'dataset_name': dataset_names, 'TV_Uniform': tv_uniform_list})
    dataset_norms_map = df_meta.groupby('dataset_name')['TV_Uniform'].mean().to_dict()
    
    print("Dataset Norms (TV Uniform):")
    for k, v in dataset_norms_map.items():
        print(f"  {k}: {v:.4f}")

    # Build List of Model Matrices
    L_matrices = []
    file_names = []

    for f_path in files:
        fname = os.path.basename(f_path)
        file_names.append(fname)
        
        df = pd.read_pickle(f_path)
        
        if len(df) != len(df_ref):
            print(f"Warning: {fname} has {len(df)} rows, expected {len(df_ref)}.")
        
        L_i_list = []
        col_name = 'Response_Distribution'
        if col_name not in df.columns:
            print(f"Error: {col_name} not found in {fname}")
            continue

        for _, row in df.iterrows():
             human_keys = sorted(list(row['human_answer'].keys())) if isinstance(row['human_answer'], dict) else []
             probs = row[col_name]
             if isinstance(probs, list) or isinstance(probs, np.ndarray):
                 if len(probs) == len(human_keys):
                     dist_dict = dict(zip(human_keys, probs))
                     L_i_list.append(parse_distribution(dist_dict, all_keys))
                 else:
                     L_i_list.append(np.zeros(len(all_keys)))
             else:
                 L_i_list.append(parse_distribution(probs, all_keys))
        
        L_matrices.append(np.array(L_i_list))

    L_stack = np.stack(L_matrices, axis=0)
    return M, M_soft, L_stack, file_names, dataset_names, dataset_norms_map

def calculate_simbench_score(M, P, dataset_names, dataset_norms_map):
    """
    Standard SimBench calculation (compares P against M).
    Here M can be Soft or Hard depending on what we want to evaluate.
    Usually SimBench compares against soft human truth.
    """
    tv_values = 0.5 * np.sum(np.abs(M - P), axis=1)
    
    scores = []
    for i, tv in enumerate(tv_values):
        ds_name = dataset_names[i]
        norm = dataset_norms_map.get(ds_name, np.nan)
        
        if not np.isnan(norm) and norm > 0:
            score = 100 * (1 - (tv / norm))
            scores.append(score)
        else:
             scores.append(np.nan)
    
    scores = np.array(scores)
    return np.nanmean(scores)

def optimize_weights(M, M_soft, L_stack, file_names, dataset_names, dataset_norms_map):
    """
    M: Target (N, K) (Hardened)
    L_stack: Models (Num_Models, N, K)
    """
    Num_Models, N = L_stack.shape[0], L_stack.shape[1]
    
    # Flatten M to vector b
    b = M.reshape(-1)
    
    # Flatten L_stack to Matrix A
    A = np.moveaxis(L_stack, 0, -1).reshape(-1, Num_Models)
    
    print(f"Optimization problem size: Matrix A {A.shape}, Vector b {b.shape}")
    print("Setting up convex optimization problem...")

    w = cp.Variable(Num_Models)

    # Objective: Minimize ||Aw - b||^2
    objective = cp.Minimize(cp.sum_squares(A @ w - b))

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    prob = cp.Problem(objective, constraints)
    
    print("Solving...")
    try:
        prob.solve()
    except cp.SolverError:
        print("Solver error, trying SCS...")
        prob.solve(solver=cp.SCS)

    print(f"Status: {prob.status}")
    print(f"Optimal Value (Loss): {prob.value:.6f}")
    
    final_alpha = w.value
    if final_alpha is None:
        return None

    # ---- SIMBENCH SCORE CALCULATION ----
    # Evaluate against M_soft (Standard SimBench) AND M (Hard target)
    print("\nCalculated SimBench Scores (Evaluating against SOFT human truth):")
    
    P_ensemble = np.sum(final_alpha[:, np.newaxis, np.newaxis] * L_stack, axis=0)
    ensemble_score = calculate_simbench_score(M_soft, P_ensemble, dataset_names, dataset_norms_map)
    print(f"  Ensemble Model: {ensemble_score:.2f}")

    best_single_sb_score = -np.inf
    best_single_name = ""
    for i in range(Num_Models):
        P_single = L_stack[i]
        score = calculate_simbench_score(M_soft, P_single, dataset_names, dataset_norms_map)
        if score > best_single_sb_score:
            best_single_sb_score = score
            best_single_name = file_names[i]
    print(f"  Best Single Model ({best_single_name}): {best_single_sb_score:.2f}")

    # Loss Metrics (against Hard target)
    print(f"\nOptimization Loss Metrics (Comparison with Hard Target):")
    
    w_uni = np.ones(Num_Models) / Num_Models
    loss_uni = np.sum((A @ w_uni - b)**2)
    print(f"  Uniform Weights Loss: {loss_uni:.6f}")
    
    losses = []
    for i in range(Num_Models):
        model_loss = np.sum((A[:, i] - b)**2)
        losses.append(model_loss)
    best_loss_idx = np.argmin(losses)
    print(f"  Best Single Model ({file_names[best_loss_idx]}) Loss: {losses[best_loss_idx]:.6f}")

    print("\nOptimal Weights:")
    if final_alpha is None:
        print("Optimization failed to return a value.")
        return None

    sorted_indices = np.argsort(final_alpha)[::-1]
    
    results = []
    for idx in sorted_indices:
        weight = final_alpha[idx]
        if weight > 0.001: 
            print(f"{file_names[idx]}: {weight:.4f}")
            results.append((file_names[idx], weight))
            
    return final_alpha

if __name__ == "__main__":
    results_dir = "results"
    
    data = load_data(results_dir)
    if data is not None and data[0] is not None:
        M_hard, M_soft, L_stack, file_names, dataset_names, dataset_norms_map = data
        
        best_weights = optimize_weights(M_hard, M_soft, L_stack, file_names, dataset_names, dataset_norms_map)
        
        if best_weights is not None:
            out_df = pd.DataFrame({
                'Model_File': file_names,
                'Weight': best_weights
            })
            out_path = os.path.join(results_dir, "optimized_convex_weights_hard.csv")
            out_df.to_csv(out_path, index=False)
            print(f"Weights saved to {out_path}")
