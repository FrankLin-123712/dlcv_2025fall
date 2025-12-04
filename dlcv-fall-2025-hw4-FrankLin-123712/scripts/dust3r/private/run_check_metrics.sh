#!/bin/bash
# #######################################  UPDATE THIS BLOCK ONLY #######################################

# --- 1. SET YOUR PATHS HERE ---
# Path to the index .txt file (defines pairs to process)
INDEX_TXT_PATH="/home/kennethyang/DLCV_hw4_1/hw_4_1_data/private_yaw_80.0_to_90.0.txt"

# Path to the ground truth .npy file
GT_NPY_PATH=/home/kennethyang/3D_Benchmark/metadata/metadata/Cambridge_Landmarks/selp_test_set.npy

# Path to the .npy file you saved from the inference script
PRED_POSE_PATH=/home/kennethyang/DLCV_hw4_1/results/private/PAIR_dust3r_predicted_poses.npy


# #######################################  UPDATE THIS BLOCK ONLY #######################################


# --- 2. SET OPTIONS ---
# Which metrics to calculate ('R', 'T', 'both')
EVAL_MODE='R'

# --- 3. EXECUTE THE SCRIPT ---
echo "Running metric check..."

python eval.py \
    --index_txt_path "${INDEX_TXT_PATH}" \
    --gt_npy_path "${GT_NPY_PATH}" \
    --pred_pose_path "${PRED_POSE_PATH}" \
    --eval_mode ${EVAL_MODE}

echo "Metric check finished."