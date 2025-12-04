import os
import sys
import torch
import numpy as np
import logging
import warnings
import gc
from tqdm import tqdm
import glob
from utils import closed_form_inverse_se3

# --- Dust3R imports & Local Imports ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DUST3R_PARENT_DIR = os.path.join(SCRIPT_DIR, 'dust3r')
if DUST3R_PARENT_DIR not in sys.path:
    sys.path.append(DUST3R_PARENT_DIR)



# --- Dust3R imports & Local Imports ---
# Now, all imports should be from 'dust3r.xxx' instead of 'dust3r.dust3r.xxx'
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


from utils import get_c2w_rotation_from_gt_dict, get_c2w_translation_from_gt_dict



from args import parse_args
from metrics import se3_to_relative_pose_error, print_summary_report # Renamed function
from utils import set_random_seeds
# --- Global Setup ---
# ... (Keep logging/warnings/precision setup) ...


def load_dust3r_model(device, model_path_arg):
    print(f"Initializing Dust3R model from {model_path_arg}...")
    model_name = model_path_arg
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    
    return model


import os
import sys
import torch
import numpy as np
import logging
import warnings
import gc
from tqdm import tqdm
import glob

def load_samples_from_files(index_txt_path, gt_npy_path, data_root, interpolated_dir, use_original_endpoints, test_only=False):
    """
    Loads sample info. Returns a list of dicts. Each dict contains:
    'scene_name': str
    'image_paths': list[str] (list of image file paths to load)
    'gt': dict (ground truth data, or None)
    """
    samples = []
    gt_data_dict = {}  # Initialize as empty dict

    # 1. Load Ground Truth Data (only if not in test_only mode)
    if not test_only:
        print(f"Loading Ground Truth from: {gt_npy_path}")
        if not gt_npy_path or not os.path.exists(gt_npy_path):
             raise FileNotFoundError(f"GT .npy file not found: {gt_npy_path}. (Use --test_only if no GT is available)")
        try:
            gt_data_dict = np.load(gt_npy_path, allow_pickle=True).item()
            print(f"Loaded {len(gt_data_dict)} GT entries.")
        except Exception as e: raise IOError(f"Could not load or parse .npy file: {e}")
    else:
        print("Running in --test_only mode. Ground truth will not be loaded.")


    # 2. Parse the Index TXT File
    print(f"Loading samples from: {index_txt_path}")
    if not os.path.exists(index_txt_path): raise FileNotFoundError(f"Index .txt file not found: {index_txt_path}")

    with open(index_txt_path, 'r') as f: lines = f.readlines()
    data_started = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('idx img1 img2'): data_started = True; continue
        if not data_started or line.startswith('---'): continue

        parts = line.split()
        if len(parts) >= 3:
            idx_key_str = parts[0]
            try:
                # --- THIS IS THE CORRECT LOGIC ---
                idx_key_int = int(idx_key_str)
                
                # Get GT data. This will be None if:
                # 1. test_only=True (because gt_data_dict is empty)
                # 2. test_only=False and the key is missing from the loaded dict
                gt_data_for_sample = gt_data_dict.get(idx_key_int)
                
                # Only warn if we *expected* GT but didn't find it.
                if not test_only and gt_data_for_sample is None:
                    print(f"Warning: idx {idx_key_str} from .txt not found in .npy GT. Proceeding without GT for this sample.")

                # --- The 'continue' is GONE. We now load images for EVERY sample. ---

                image_paths = [] # List of paths for this sample
                original_img1_rel_path = parts[1]
                original_img2_rel_path = parts[2]

                # --- Determine list of image paths ---
                if use_original_endpoints:
                    # Mode 3: Combined
                    if not data_root or not interpolated_dir:
                            raise ValueError("--use_original_endpoints requires both --data_root and --interpolated_dir")
                    img1_path = os.path.join(data_root, original_img1_rel_path)
                    img2_path = os.path.join(data_root, original_img2_rel_path)
                    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                        raise FileNotFoundError(f"Original endpoint image not found: {img1_path} or {img2_path}")

                    scene_img_dir = os.path.join(interpolated_dir, idx_key_str, 'dynamicrafter')
                    if not os.path.isdir(scene_img_dir):
                            raise FileNotFoundError(f"Interpolated dir not found: {scene_img_dir}")
                    all_pngs = glob.glob(os.path.join(scene_img_dir, "*.png"))
                    def get_frame_num(p): 
                        try: return int(os.path.basename(p).split('frame')[-1].split('.')[0]) 
                        except: return -1
                    sorted_pngs = sorted(all_pngs, key=get_frame_num)

                    if len(sorted_pngs) < 3: 
                        raise ValueError(f"Expected >= 3 frames in {scene_img_dir}, found {len(sorted_pngs)}")
                    interpolated_middle_frames = sorted_pngs[1:-1]
                    image_paths = [img1_path] + interpolated_middle_frames + [img2_path]
                    # print(f"  idx {idx_key_str}: Using combined mode - {len(image_paths)} frames.") # Optional: uncomment for verbose logging

                elif interpolated_dir:
                    # Mode 2: Interpolated Only
                    scene_img_dir = os.path.join(interpolated_dir, idx_key_str, 'dynamicrafter')
                    if not os.path.isdir(scene_img_dir):
                        raise FileNotFoundError(f"Interpolated dir not found: {scene_img_dir}")
                    all_pngs = glob.glob(os.path.join(scene_img_dir, "*.png"))
                    def get_frame_num(p): 
                        try: return int(os.path.basename(p).split('frame')[-1].split('.')[0]) 
                        except: return -1
                    sorted_pngs = sorted(all_pngs, key=get_frame_num)
                    if len(sorted_pngs) < 2: raise ValueError(f"Expected >= 2 frames in {scene_img_dir}")
                    image_paths = [sorted_pngs[:]] # Only first and last
                    # print(f"  idx {idx_key_str}: Using interpolated endpoints mode - {len(image_paths)} frames.") # Optional: uncomment for verbose logging

                elif data_root:
                    # Mode 1: Original Only
                    img1_path = os.path.join(data_root, original_img1_rel_path)
                    img2_path = os.path.join(data_root, original_img2_rel_path)
                    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                            raise FileNotFoundError(f"Original image not found: {img1_path} or {img2_path}")
                    image_paths = [img1_path, img2_path]
                    # print(f"  idx {idx_key_str}: Using original mode - {len(image_paths)} frames.") # Optional: uncomment for verbose logging
                else:
                    raise ValueError("Logic error: No valid image source configuration.")
                
                # Add the sample (gt_data_for_sample will be None if not found or in test_only)
                samples.append({
                    'scene_name': idx_key_str,
                    'image_paths': image_paths, 
                    'gt': gt_data_for_sample 
                })
                # --- END CORRECT LOGIC ---

            except (ValueError, FileNotFoundError, Exception) as e:
                # Catch errors during path construction or existence check for one line
                print(f"Warning: Error processing line for idx '{idx_key_str}': {e}. Skipping sample.")

    print(f"Loaded {len(samples)} valid samples to process.")
    if len(samples) == 0:
        raise ValueError("No valid samples loaded. Check file paths and content.")
    return samples


def read_gt_dict(gt_raw):
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
    gt_c2w_se3_pair = torch.cat([gt_rot_tensor, gt_tvec_tensor], dim=2)        

    return gt_c2w_se3_pair



def inference_one_scene(model, sample, device, args):
    """
    Performs inference for one scene/sample using Dust3R model.
    sample: dict with keys 'scene_name', 'image_paths', 'gt'
    Returns predicted extrinsics as torch.Tensor of shape (S, 3, 4)
    """
    image_paths = sample['image_paths']
    scene_name = sample['scene_name']
    image_paths = sample['image_paths'] # Now a list of paths
    gt_data_raw = sample['gt']

    # Dust3r inference
    # TODO ===============================================================================
    images = load_images(image_paths, size=args.dust3r_image_size)  # (S, C, H, W)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    if len(images)<=2:
        """ run for two images """
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)
    else:
        """ run for a sequence of images """
        
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

        _ = scene.compute_global_alignment(
            init="mst", niter=args.dust3r_align_niter,
            schedule=args.dust3r_align_schedule, lr=args.dust3r_align_lr
        )
    # ===============================================================================



    # Extract Aligned Poses (C2W)
    poses_from_scene = scene.get_im_poses() # List of N tensors (N = len(image_paths))
    poses_np_cpu = [p.detach().cpu().numpy() for p in poses_from_scene]

    # Check expected number of poses ---
    expected_num_poses = len(image_paths)
    if len(poses_np_cpu) != expected_num_poses:
        raise ValueError(
                    f"Expected {expected_num_poses} poses from DUST3R for {scene_name}, "
                    f"but got {len(poses_np_cpu)}."
                )
    
    
    # Store all poses, but metrics/plotting will only use first/last
    pred_extrinsic = np.stack(poses_np_cpu, axis=0) # Shape (S, 4, 4)
    
    ### --- RENAMED: gt_pose is C2W --- ###
    gt_pose = read_gt_dict(gt_data_raw) 

    return pred_extrinsic, gt_pose


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_random_seeds(args.seed)
    model = load_dust3r_model(device, args.model_path)
    
    samples_to_process = load_samples_from_files(
        args.index_txt_path, args.gt_npy_path, args.data_root, args.interpolated_dir, args.use_original_endpoints, args.test_only
    )


    all_r_errors = []
    all_t_errors = []
    all_predicted_poses = {}

    for sample in tqdm(samples_to_process, desc="Processing Samples"):
        pred_extrinsic, gt_c2w_se3_pair = inference_one_scene(model, sample, device, args)
        pred_extrinsic = torch.from_numpy(pred_extrinsic).to(device)

        # -----------------------------------------------------------------
        # --- 1. PREDICTION:  4x4 C2W ---
        # -----------------------------------------------------------------
        # Select first and last predicted C2W SE(3) poses
        pred_c2w_pair = pred_extrinsic[[0, -1]] # (2, 4, 4) C2W

        # -----------------------------------------------------------------
        ### --- 2. GROUND TRUTH: Convert GT 3x4 C2W to 4x4 C2W --- ###
        # -----------------------------------------------------------------

        if gt_c2w_se3_pair is None:
            # Create a (2, 3, 4) tensor full of NaNs
            # Use the prediction's dtype and device for compatibility
            gt_c2w_se3_pair = torch.full((2, 3, 4), float('nan'), device=device, dtype=pred_extrinsic.dtype)
        else:
            # Move existing GT to device
            gt_c2w_se3_pair = gt_c2w_se3_pair.to(device)



        add_row_gt = torch.tensor([0, 0, 0, 1], device=device, dtype=gt_c2w_se3_pair.dtype).expand(2, 1, 4)
        gt_c2w_4x4_pair = torch.cat((gt_c2w_se3_pair, add_row_gt), dim=1) # (2, 4, 4) C2W

        # -----------------------------------------------------------------
        ### --- 3. Compare C2W vs C2W --- ###
        # -----------------------------------------------------------------
        # GT is C2W. DUST3R prediction is C2W.
        #         
        # This is the (2, 4, 4) C2W prediction
        pred_to_compare = pred_c2w_pair 
        scene_idx_str = sample['scene_name']
        all_predicted_poses[scene_idx_str] = pred_to_compare.detach().cpu().numpy()
        
        # This is the (2, 4, 4) C2W ground truth
        gt_to_compare = gt_c2w_4x4_pair 

        # -----------------------------------------------------------------
        
        # Now both inputs are C2W
        r_err, t_err = se3_to_relative_pose_error(
            pred_to_compare,    # (2, 4, 4) C2W
            gt_to_compare,      # (2, 4, 4) C2W
            2
        )
    
        all_r_errors.append(r_err.item())
        all_t_errors.append(t_err.item())


    ### --- Save predicted poses dictionary after the loop --- ###
    if args.save_pose_path:
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(args.save_pose_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            print(f"\nSaving {len(all_predicted_poses)} predicted poses to {args.save_pose_path}...")
            np.save(args.save_pose_path, all_predicted_poses)
            print("Poses saved successfully.")
        
        except Exception as e:
            print(f"\nWarning: Could not save predicted poses to {args.save_pose_path}. Error: {e}")

    print_summary_report(all_r_errors, all_t_errors, args.eval_mode)
    print(f"\nEvaluation finished. Model: {args.model_path}")


if __name__ == "__main__":
    # (Dependency checks)
    for pkg in ['numpy', 'torch', 'scipy', 'tqdm']: # Added tqdm
        try: __import__(pkg)
        except ImportError: print(f"Error: Dependency '{pkg}' not found."); sys.exit(1)
    args = parse_args()
    main()