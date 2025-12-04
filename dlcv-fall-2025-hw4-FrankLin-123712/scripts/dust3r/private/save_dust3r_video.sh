#!/bin/bash

# #######################################  UPDATE THIS BLOCK ONLY #######################################
# --- Path to save the predicted poses ---
SAVE_POSE_PATH="results/private/VIDEO_dust3r_predicted_poses.npy"
TEST_ONLY_ARG="--test_only"  # Add this flag to only run inference and save poses without evaluation


# --- 1. SET YOUR PATHS HERE ---
# Path to the index .txt file (defines pairs to process)
INDEX_TXT_PATH="/home/kennethyang/DLCV_hw4_1/hw_4_1_data/private_yaw_80.0_to_90.0.txt"


# Root directory containing the ORIGINAL images (e.g., Street/, GreatCourt/)
DATA_ROOT="/home/kennethyang/DLCV_hw4_1/hw_4_1_data/private/images"
INTERPOLATED_DIR="/home/kennethyang/DLCV_hw4_1/hw_4_1_data/private/interpolated_images/results_wide_baseline/dynamicrafter_512_wide_baseline_seed12306" 
MODEL_PATH="/home/kennethyang/ExtremeVideoPose/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# #####################################################################################################################


USE_MODEL="Dust3R"  # Change to "Dust3R" to run Dust3R evaluation


# --- 2. SET OPTIONS ---
# Which metrics to calculate ('R', 'T', 'both') - T requires GT tx,ty,tz
EVAL_MODE='R'

# Batch size for DataLoader
BATCH_SIZE=8
NUM_WORKERS=4

# Random Seed (Don't modify)
SEED=0


# --- 3. EXECUTE THE SCRIPT ---
echo "Running Dust3r evaluation on ORIGINAL image pairs using Co3D-style W2C metrics..."



# Construct model path argument
MODEL_PATH_ARG=""
if [[ -n "${MODEL_PATH}" ]]; then
  MODEL_PATH_ARG="--model_path ${MODEL_PATH}"
fi

# Run the VGGT inference script (assuming it's in the 'videopose' subdir relative to this script)
# If script is in co3d_eval, adjust path e.g., python co3d_eval/vggt_inference_co3d.py
# Assuming this bash script is in ExtremeVideoPose and python script is in ExtremeVideoPose/videopose
python dust3r_inference.py \
    --index_txt_path "${INDEX_TXT_PATH}" \
    --gt_npy_path "${GT_NPY_PATH}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --eval_mode ${EVAL_MODE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --use_original_endpoints \
    --interpolated_dir "${INTERPOLATED_DIR}" \
    ${MODEL_PATH_ARG} \
    --use_model ${USE_MODEL} \
    --save_pose_path "${SAVE_POSE_PATH}" \
    ${TEST_ONLY_ARG} # <--- ADDED THIS LINE