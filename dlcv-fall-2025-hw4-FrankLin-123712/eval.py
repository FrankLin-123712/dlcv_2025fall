import os
import sys
import torch
import numpy as np
import logging
from tqdm import tqdm
import argparse  # <-- Import argparse directly

# Import components from your other scripts
from metrics import se3_to_relative_pose_error, print_summary_report
from utils import get_c2w_rotation_from_gt_dict, get_c2w_translation_from_gt_dict

def parse_args():
    """
    Parses arguments specifically for the check_metrics script.
    """
    parser = argparse.ArgumentParser(description="Evaluate saved pose predictions against ground truth.")
    
    parser.add_argument('--index_txt_path', type=str, required=True,
                        help="Path to the .txt file listing samples to evaluate.")
    
    parser.add_argument('--gt_npy_path', type=str, required=True,
                        help="Path to the ground truth .npy file.")
    
    parser.add_argument('--pred_pose_path', type=str, required=True,
                        help="Path to the saved predicted poses .npy file (e.g., 'results/dust3r_poses.npy')")
    
    parser.add_argument('--eval_mode', type=str, default='R', choices=['R', 'T', 'both'],
                        help="Which metrics to calculate ('R', 'T', 'both').")
    
    return parser.parse_args()


# --- This function is COPIED from dust3r_inference.py ---
# We need it here to process the raw GT data
def read_gt_dict(gt_raw):
    """
    Parses a raw GT dictionary entry into a (2, 3, 4) C2W tensor.
    """
    if gt_raw is None:
        return None
        
    # --- Extract GT Rotations (C2W) ---
    gt_c2w_rot1 = get_c2w_rotation_from_gt_dict(gt_raw['img1']) 
    gt_c2w_rot2 = get_c2w_rotation_from_gt_dict(gt_raw['img2']) 
   
    gt_c2w_rot_pair = np.stack([gt_c2w_rot1, gt_c2w_rot2], axis=0) # (2, 3, 3)

    # --- Extract GT Translations (C2W) if possible ---
    gt_c2w_tvec1 = get_c2w_translation_from_gt_dict(gt_raw['img1'], gt_c2w_rot1) 
    gt_c2w_tvec2 = get_c2w_translation_from_gt_dict(gt_raw['img2'], gt_c2w_rot2) 

    # Store as a pair (can contain None if keys were missing)
    gt_c2w_tvec_pair = None
    if gt_c2w_tvec1 is not None and gt_c2w_tvec2 is not None:
        gt_c2w_tvec_pair = np.stack([gt_c2w_tvec1, gt_c2w_tvec2], axis=0) # (2, 3)

    # Convert numpy arrays to tensors for collation
    gt_rot_tensor = torch.from_numpy(gt_c2w_rot_pair).to(torch.float64)   # (2, 3, 3) 
    # Handle potential None for translation
    gt_tvec_tensor = torch.from_numpy(gt_c2w_tvec_pair).to(torch.float64) \
                        if gt_c2w_tvec_pair is not None else torch.full((2, 3), float('nan'))

    
    # --- Store the full 4x4 C2W GT matrix pair ---
    # 1. Reshape tvec from (2, 3) to (2, 3, 1) so it can be concatenated
    gt_tvec_tensor  = gt_tvec_tensor.unsqueeze(-1)
    gt_c2w_se3_pair = torch.cat([gt_rot_tensor, gt_tvec_tensor], dim=2) # (2, 3, 4)    

    return gt_c2w_se3_pair
# --- End of copied function ---


def load_index_list(index_txt_path):
    """
    Parses the .txt file and returns a list of integer keys (IDs) to process.
    """
    print(f"Loading sample index from: {index_txt_path}")
    if not os.path.exists(index_txt_path): 
        raise FileNotFoundError(f"Index .txt file not found: {index_txt_path}")
    
    sample_keys = []
    with open(index_txt_path, 'r') as f: 
        lines = f.readlines()
        
    data_started = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('idx img1 img2'): 
            data_started = True; 
            continue
        if not data_started or line.startswith('---'): 
            continue

        parts = line.split()
        if len(parts) >= 3:
            try:
                idx_key_int = int(parts[0])
                sample_keys.append(idx_key_int)
            except ValueError:
                print(f"Warning: Skipping malformed line in index file: {line}")
                
    print(f"Found {len(sample_keys)} samples to evaluate in index file.")
    return sample_keys


def main():
    args = parse_args() # <-- Calls the local parse_args() function
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Predicted Poses
    print(f"Loading predicted poses from: {args.pred_pose_path}")
    if not args.pred_pose_path or not os.path.exists(args.pred_pose_path):
        raise FileNotFoundError(f"Predicted poses file not found: {args.pred_pose_path}")
    pred_pose_dict = np.load(args.pred_pose_path, allow_pickle=True).item()
    print(f"Loaded {len(pred_pose_dict)} predicted poses.")

    # 2. Load Ground Truth Data
    print(f"Loading Ground Truth from: {args.gt_npy_path}")
    if not args.gt_npy_path or not os.path.exists(args.gt_npy_path):
        raise FileNotFoundError(f"GT .npy file not found: {args.gt_npy_path}")
    gt_data_dict = np.load(args.gt_npy_path, allow_pickle=True).item()
    print(f"Loaded {len(gt_data_dict)} GT entries.")
    
    # 3. Load Index of samples to process
    sample_keys_to_process = load_index_list(args.index_txt_path)

    all_r_errors = []
    all_t_errors = []

    print(f"Available keys in pred_pose_dict are: {list(pred_pose_dict.keys())}")
    print(f"Available keys in GT_pose_dict are: {list(gt_data_dict.keys())}")
    
    print("\nStarting metric calculation...")
    for key_int in tqdm(sample_keys_to_process, desc="Evaluating Samples"):
        key_str = str(key_int)
        
        # --- Check if we have both GT and Prediction for this sample ---
        if key_int not in gt_data_dict:
            print(f"Warning: Skipping key {key_int} (from .txt), not found in GT .npy file.")
            continue
        if key_str not in pred_pose_dict:
            print(f"Warning: Skipping key {key_int} (from .txt), not found in predictions .npy file.")
            continue
            
        # --- We have both, let's process them ---
        
        # 1. Get raw GT data and process it
        gt_raw = gt_data_dict[key_int]
        gt_c2w_se3_pair = read_gt_dict(gt_raw) # (2, 3, 4) tensor
        
        if gt_c2w_se3_pair is None:
             print(f"Warning: Skipping key {key_int}, failed to read GT data.")
             continue
        
        gt_c2w_se3_pair = gt_c2w_se3_pair.to(device)
        add_row_gt = torch.tensor([0, 0, 0, 1], device=device, dtype=gt_c2w_se3_pair.dtype).expand(2, 1, 4)
        gt_c2w_4x4_pair = torch.cat((gt_c2w_se3_pair, add_row_gt), dim=1) # (2, 4, 4) C2W
        
        # 2. Get predicted poses (already (2, 4, 4) C2W)
        pred_c2w_pair_np = pred_pose_dict[key_str]
        pred_c2w_pair_tensor = torch.from_numpy(pred_c2w_pair_np).to(device) # (2, 4, 4) C2W
        
        # 3. Compare C2W vs C2W
        pred_to_compare = pred_c2w_pair_tensor
        gt_to_compare = gt_c2w_4x4_pair
        
        r_err, t_err = se3_to_relative_pose_error(
            pred_to_compare,    # (2, 4, 4) C2W
            gt_to_compare,      # (2, 4, 4) C2W
            2
        )
    
        all_r_errors.append(r_err.item())
        all_t_errors.append(t_err.item())

    
    # --- Print final report ---
    if not all_r_errors:
        print("\nNo samples were successfully evaluated. Check your input files and paths.")
    else:
        print_summary_report(all_r_errors, all_t_errors, args.eval_mode)
    
    print(f"\nMetric calculation finished.")


if __name__ == "__main__":
    # (Dependency checks)
    for pkg in ['numpy', 'torch', 'tqdm']:
        try: __import__(pkg)
        except ImportError: print(f"Error: Dependency '{pkg}' not found."); sys.exit(1)
    
    # Ensure all required utils are available
    try:
        from metrics import se3_to_relative_pose_error, print_summary_report
        from utils import get_c2w_rotation_from_gt_dict, get_c2w_translation_from_gt_dict
    except ImportError as e:
        print(f"Error: Missing required import. Make sure 'metrics.py', and 'utils.py' are accessible.")
        print(f"Details: {e}")
        sys.exit(1)
        
    main()