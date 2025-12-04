# /home/kennethyang/ExtremeVideoPose/videopose/metrics.py
import torch
import numpy as np
import warnings


from utils import build_pair_index, mat_to_quat, closed_form_inverse_se3


# --- Core Error Calculation Functions (W2C convention, from test_co3d.py) ---
# rotation_angle and translation_angle remain the same as previous version
# ... (Keep rotation_angle and translation_angle implementations) ...
def rotation_angle(rot_gt, rot_pred, eps=1e-7):
    """Calculates geodesic rotation angle error in degrees."""
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    dot_product = torch.abs((q_pred * q_gt).sum(dim=-1))
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
    err_rad = 2 * torch.acos(dot_product)
    return err_rad * 180 / np.pi

def compare_translation_by_angle(t_gt, t, eps=1e-7):
    """Normalize vectors and compute angle between them (radians)."""
    t_norm = torch.norm(t, dim=-1, keepdim=True)
    t_gt_norm = torch.norm(t_gt, dim=-1, keepdim=True)
    valid_mask = (t_norm > eps) & (t_gt_norm > eps)
    err_t = torch.full_like(t_norm.squeeze(-1), np.pi)
    if valid_mask.any():
        t_valid = t[valid_mask] / t_norm[valid_mask]
        t_gt_valid = t_gt[valid_mask] / t_gt_norm[valid_mask]
        dot_prod = (t_valid * t_gt_valid).sum(dim=-1)
        dot_prod = torch.clamp(dot_prod, -1.0 + eps, 1.0 - eps)
        err_t[valid_mask.squeeze(-1)] = torch.acos(dot_prod)
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = np.pi
    return err_t

def translation_angle(tvec_gt, tvec_pred, ambiguity=True):
    """Calculate translation angle error (degrees)."""
    rel_tangle_rad = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_rad * 180.0 / np.pi
    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, 180.0 - rel_tangle_deg)
    return rel_tangle_deg





def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function ASSUMES the input poses are world-to-camera (w2c) transformations.
    ...
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # --- Start: NaN Handling ---
    gt_trans = gt_se3[:, :3, 3]
    has_nan_gt_trans = torch.isnan(gt_trans).any()

    if has_nan_gt_trans:
        gt_se3_clean = gt_se3.clone()
        gt_se3_clean[:, :3, 3] = torch.nan_to_num(gt_trans, nan=0.0)
    else:
        gt_se3_clean = gt_se3
    # --- End: NaN Handling ---
    
    relative_pose_gt = gt_se3_clean[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3_clean[pair_idx_i2])
    )
    
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )

    if has_nan_gt_trans:
        rel_tangle_deg = torch.full_like(rel_rangle_deg, float('nan'))
    else:
        rel_tangle_deg = translation_angle(
            relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
        )


    print(f"DEBUG: Rotation Error (°): {rel_rangle_deg}")
    print(f"DEBUG: Translation Error (°): {rel_tangle_deg}")

    return rel_rangle_deg, rel_tangle_deg





# --- FINAL CORRECTED Error Calculation Function (C2W) ---
def calculate_relative_pose_error_c2w(pred_c2w_pair, gt_c2w_rot_pair, gt_c2w_tvec_pair, eval_mode):
    """
    Computes relative pose errors assuming all inputs are C2W (Cam2World).
    Assumes inputs are:
        pred_c2w_pair: (2, 4, 4) Predicted C2W SE(3) poses for start and end.
        gt_c2w_rot_pair: (2, 3, 3) GT C2W Rotations for start and end.
        gt_c2w_tvec_pair: (2, 3) GT C2W Translations for start and end.
    """
    r_err = torch.tensor(np.nan, device=pred_c2w_pair.device, dtype=pred_c2w_pair.dtype)
    t_err = torch.tensor(np.nan, device=pred_c2w_pair.device, dtype=pred_c2w_pair.dtype)

    try:
        # --- 1. Predicted C2W Relative Pose ---
        # P_rel = inv(P_start) @ P_end
        R_pred_start = pred_c2w_pair[0, :3, :3]
        t_pred_start = pred_c2w_pair[0, :3, 3]
        R_pred_end = pred_c2w_pair[1, :3, :3]
        t_pred_end = pred_c2w_pair[1, :3, 3]

        # R_rel = R_start.T @ R_end
        pred_rot_rel = R_pred_start.T @ R_pred_end
        # t_rel = R_start.T @ (t_end - t_start)
        pred_tvec_rel = R_pred_start.T @ (t_pred_end - t_pred_start)


        # --- 2. Ground Truth C2W Relative Pose ---
        R_gt_start = gt_c2w_rot_pair[0]
        t_gt_start = gt_c2w_tvec_pair[0]
        R_gt_end = gt_c2w_rot_pair[1]
        t_gt_end = gt_c2w_tvec_pair[1]

        # --- Rotation Error Calculation ---
        if eval_mode in ['R', 'both']:
            # R_rel = R_start.T @ R_end
            gt_rot_rel = R_gt_start.T @ R_gt_end
            
            # Compare relative rotations
            r_err = rotation_angle(gt_rot_rel.unsqueeze(0), pred_rot_rel.unsqueeze(0)).squeeze()

        # --- Translation Error Calculation ---
        if eval_mode in ['T', 'both']:
            # Check if GT translation is valid (not NaN)
            if not torch.isnan(gt_c2w_tvec_pair).any():
                # t_rel = R_start.T @ (t_end - t_start)
                gt_tvec_rel = R_gt_start.T @ (t_gt_end - t_gt_start)
                
                # Compare relative translation vectors
                t_err = translation_angle(gt_tvec_rel.unsqueeze(0), pred_tvec_rel.unsqueeze(0)).squeeze()
            # else: t_err remains NaN

    except Exception as e:
        warnings.warn(f"Error during relative pose error calculation: {e}")

    return r_err, t_err
# --- END CORRECTION ---









def calculate_relative_error_c2w_inputs(pred_c2w_pair, gt_c2w_pair):
    """
    Wrapper to calculate relative pose error from C2W (Camera-to-World) inputs.
    
    This function first converts the C2W poses to W2C (World-to-Camera)
    before passing them to the 'se3_to_relative_pose_error' function,
    which expects W2C inputs.

    Args:
        pred_c2w_pair: (2, 4, 4) tensor of predicted C2W poses [start, end]
        gt_c2w_pair: (2, 4, 4) tensor of ground truth C2W poses [start, end]
                       (Can contain NaNs in the translation part)

    Returns:
        (r_err, t_err): Rotation and translation errors (tensors)
    """
    
    # 1. Convert C2W -> W2C
    # The inverse of a C2W pose is the W2C pose
    pred_w2c_pair = closed_form_inverse_se3(pred_c2w_pair)
    gt_w2c_pair = closed_form_inverse_se3(gt_c2w_pair)
    
    # 2. Call the original W2C error function
    # We pass num_frames=2, which 'build_pair_index' will use
    # to create the single pair (0, 1).
    r_err, t_err = se3_to_relative_pose_error(
        pred_w2c_pair, 
        gt_w2c_pair, 
        2
    )
    
    # r_err and t_err will be tensors of shape (1,)
    return r_err, t_err















# --- AUC Calculation and Summary (Minor adjustments for clarity) ---

def calculate_auc_np(r_errors_np, t_errors_np, eval_mode, max_threshold=30):
    """Calculates AUC based on specified errors."""
    valid_r = ~np.isnan(r_errors_np)
    valid_t = ~np.isnan(t_errors_np)

    if eval_mode == 'R':
        if not np.any(valid_r): return 0.0
        errors_to_use = np.abs(r_errors_np[valid_r])
    elif eval_mode == 'T':
        if not np.any(valid_t): return 0.0
        errors_to_use = np.abs(t_errors_np[valid_t])
    elif eval_mode == 'both':
        valid_both = valid_r & valid_t
        if not np.any(valid_both): return 0.0
        errors_to_use = np.maximum(np.abs(r_errors_np[valid_both]), np.abs(t_errors_np[valid_both]))
    else: # Should not happen
        return 0.0

    if len(errors_to_use) == 0: return 0.0

    accuracies = []
    for th in range(1, max_threshold + 1):
        accuracies.append(np.mean(errors_to_use < th))

    return np.mean(accuracies) if accuracies else 0.0

def print_summary_report(all_r_errors, all_t_errors, eval_mode):
    """Prints a formatted summary of evaluation results."""
    # (Implementation remains largely the same, but uses the modified AUC calc)
    # ... [Keep implementation from previous version, just ensure calculate_auc_np is called correctly] ...
    BOLD = "\033[1m"; RED = "\033[91m"; RESET = "\033[0m"; BLUE = "\033[94m"; GREEN="\033[92m"
    print("\n" + "="*80)
    print(f"{BOLD}Evaluation Summary (Mode: {eval_mode}){RESET}")
    print("-"*60)
    r_np, t_np = np.array(all_r_errors), np.array(all_t_errors)
    valid_r = r_np[~np.isnan(r_np)]; valid_t = t_np[~np.isnan(t_np)]
    num_total = len(all_r_errors)
    print(f"Total samples processed: {num_total}")

    if eval_mode in ['R', 'both']:
        if len(valid_r) > 0:
            print(f"\nRotation Errors - {len(valid_r)} valid samples:")
            print(f"  Mean (MRE):   {np.mean(valid_r):.3f}°")
            for th in [5, 15, 30]: print(f"    Acc < {th}°:   {np.mean(valid_r < th) * 100:.2f}%")
        else: print("\nNo valid Rotation Errors calculated.")

    if eval_mode in ['T', 'both']:
        if len(valid_t) > 0:
            print(f"\nTranslation Angle Errors - {len(valid_t)} valid samples:")
            print(f"  Mean (MTE):   {np.mean(valid_t):.3f}°")
            for th in [5, 15, 30]: print(f"    Acc < {th}°:   {np.mean(valid_t < th) * 100:.2f}%")
        else:
            from videopose.utils import _warned_missing_translation_gt # Check flag
            if _warned_missing_translation_gt: print("\nTranslation Errors skipped due to missing GT 'tx','ty','tz'.")
            else: print("\nNo valid Translation Angle Errors calculated.")

    # AUC Summary (based on eval_mode)
    print(f"\n{BOLD}AUC ({'Max(R,T)' if eval_mode=='both' else eval_mode}) @ Threshold:{RESET}")
    for th in [30, 15, 5, 3]:
        auc = calculate_auc_np(r_np, t_np, eval_mode, max_threshold=th)
        print(f"  {RED}AUC @ {th}°:{RESET}    {auc:.4f}")
    print("="*80)