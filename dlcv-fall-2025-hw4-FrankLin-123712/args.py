import argparse
import os

def parse_args():
    """Parses command-line arguments for the VGGT evaluation script."""
    parser = argparse.ArgumentParser(description="VGGT Pose Estimation using W2C Relative Pose Metrics")

    # --- Paths ---
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a local VGGT model checkpoint (.pt). If not set, downloads from HuggingFace.')
    parser.add_argument('--index_txt_path', type=str, required=True,
                        help="Path to the .txt file defining image pairs.")
    parser.add_argument('--gt_npy_path', type=str, required=True,
                        help="Path to the .npy file with GT poses (assumed C2W Quat+Center).")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Root directory for original images. This is always required.")
    parser.add_argument('--interpolated_dir', type=str, default=None,
                        help="Root directory for interpolated images. Use with --use_original_endpoints.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save evaluation results (e.g., summary text).")

    # --- Execution Flags ---
    parser.add_argument('--use_original_endpoints', action='store_true',
                        help="Combine original start/end frames with interpolated middle frames.")
    
    parser.add_argument('--use_model', type=str, choices=["VGGT", "Dust3R"], default='VGGT',
                        help="Choose which model to use for inference: 'VGGT' or 'Dust3R'.")    
    # --- ADDED --use_ba ---
    parser.add_argument('--use_ba', action='store_true', default=False,
                        help='Enable bundle adjustment (requires ba.py from VGGT eval code).')
    # --- END ADDED ---

    # --- Evaluation Parameters ---
    parser.add_argument('--eval_mode', type=str, choices=['R', 'T', 'both'], default='R',
                        help="Metrics to evaluate: 'R'(Rotation), 'T'(Translation Angle), or 'both'.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for the DataLoader.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers for the DataLoader.")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')
    
    dust3r_group = parser.add_argument_group('DUST3R Alignment Options')
    dust3r_group.add_argument( "--dust3r_align_niter", type=int, default=100, help="Iterations for DUST3R alignment.")
    dust3r_group.add_argument( "--dust3r_align_lr", type=float, default=0.01, help="LR for DUST3R alignment.")
    dust3r_group.add_argument( "--dust3r_align_schedule", type=str, default='cosine', help="Schedule for DUST3R alignment.")
    dust3r_group.add_argument( "--dust3r_image_size", type=int, default=480, help="Resize images to this size for DUST3R.")



    # Inside args.py, in your parse_args function:
    parser.add_argument('--save_pose_path', type=str, default=None, 
                        help="Path to save the predicted poses .npy file (e.g., 'results/dust3r_poses.npy')")
    # In args.py, inside parse_args():
    parser.add_argument('--test_only', action='store_true',
                    help="Run in test-only mode. No GT .npy file is required and metrics will not be calculated.")

    args = parser.parse_args()








    # --- Post-parsing validation ---
    if args.model_path and not os.path.isfile(args.model_path):
        parser.error(f"Model checkpoint not found: {args.model_path}")
    if not os.path.isdir(args.data_root):
        parser.error(f"Data root directory not found: {args.data_root}")
    if args.use_original_endpoints and not args.interpolated_dir:
        parser.error("--use_original_endpoints requires --interpolated_dir to be set.")
    
    return args