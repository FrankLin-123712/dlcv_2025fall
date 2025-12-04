# /home/kennethyang/ExtremeVideoPose/videopose/utils.py
import torch
import numpy as np
import random
import os
import sys
import warnings
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F



# --- Pose Conversion ---
_warned_missing_translation_gt = False



# --- Pose Conversion ---
_warned_missing_translation_gt = False


def get_c2w_rotation_from_gt_dict(gt_dict):
    """
    Extracts only the 3x3 W2C rotation matrix from a GT dictionary (C2W Quat).
    Returns R_w2c (numpy array) or None.
    """
    try:
        q = [gt_dict['qx'], gt_dict['qy'], gt_dict['qz'], gt_dict['qw']]
        R_c2w = R.from_quat(q).as_matrix()
        R_w2c = R_c2w.T # World-to-Camera rotation
        return R_w2c
    except KeyError as e:
        warnings.warn(f"GT dictionary missing rotation key: {e}. Cannot extract rotation.")
        return None
    except Exception as e:
        warnings.warn(f"Error extracting GT rotation: {e}")
        return None

def get_c2w_translation_from_gt_dict(gt_dict, R_w2c):
    """
    Extracts the 3x1 W2C translation vector (t_w2c = -R_w2c @ C_c2w).
    Requires the corresponding W2C rotation matrix.
    Returns t_w2c (numpy array) or None if translation keys are missing.
    """
    global _warned_missing_translation_gt
    try:
        if not all(k in gt_dict for k in ['tx', 'ty', 'tz']):
            if not _warned_missing_translation_gt:
                warnings.warn("GT data missing 'tx', 'ty', or 'tz'. Cannot calculate W2C translation.", UserWarning)
                _warned_missing_translation_gt = True
            return None # Cannot form W2C translation without Center C

        C_c2w = np.array([gt_dict['tx'], gt_dict['ty'], gt_dict['tz']])
        t_w2c = -R_w2c @ C_c2w # Calculate W2C translation
        return t_w2c

    except Exception as e:
        warnings.warn(f"Error calculating GT W2C translation: {e}")
        return None




def convert_gt_dict_to_w2c_se3(gt_dict):
    """
    Converts a GT dictionary (C2W Quat+Center) to a 4x4 W2C SE(3) numpy matrix.
    Returns None if essential keys are missing OR if translation is needed but absent.
    """
    global _warned_missing_translation_gt
    try:
        # 1. Extract C2W Rotation (R_c2w)
        q = [gt_dict['qx'], gt_dict['qy'], gt_dict['qz'], gt_dict['qw']]
        R_c2w = R.from_quat(q).as_matrix()

        # 2. Extract C2W Center (C_c2w) - Required for W2C conversion
        if not all(k in gt_dict for k in ['tx', 'ty', 'tz']):
            if not _warned_missing_translation_gt:
                warnings.warn("GT data missing 'tx', 'ty', or 'tz' keys. Cannot form W2C poses.", UserWarning)
                _warned_missing_translation_gt = True
            return None # Cannot form W2C without Center C

        C_c2w = np.array([gt_dict['tx'], gt_dict['ty'], gt_dict['tz']])

        # 3. Convert C2W to W2C Rotation (R_w2c) and Translation (t_w2c)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C_c2w

        # 4. Form 4x4 W2C SE(3) matrix
        se3_w2c = np.identity(4)
        se3_w2c[:3, :3] = R_w2c
        se3_w2c[:3, 3] = t_w2c
        return se3_w2c

    except KeyError as e:
        warnings.warn(f"GT dictionary missing required key: {e}. Cannot convert pose.")
        return None
    except Exception as e:
        warnings.warn(f"Error converting GT pose: {e}")
        return None

# --- Indexing (From test_co3d.py) ---
def build_pair_index(N):
    """Build indices for all possible pairs of N frames."""
    if N < 2:
        return torch.tensor([]).long(), torch.tensor([]).long() # Ensure long type
    # Ensure indices are long type for tensor indexing
    i1, i2 = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    return i1.long(), i2.long()



def get_relative_poses(poses_w2c, ref_idx):
    """
    Calculates all poses relative to a specific reference frame.

    This function assumes poses_w2c contains world-to-camera transformations.

    Args:
        poses_w2c: Tensor of N absolute world-to-camera poses, shape (N, 4, 4)
        ref_idx: The index (0 to N-1) of the frame to use as the reference.

    Returns:
        poses_ref_to_cam: Tensor of N relative poses, shape (N, 4, 4).
                          The pose at ref_idx will be the identity matrix.
    """
    # Get the reference pose (world-to-camera)
    # T_cr_w
    ref_pose_w2c = poses_w2c[ref_idx]
    
    # Get the inverse of the reference pose (camera-to-world)
    # (T_cr_w)^-1 = T_w_cr
    ref_pose_c2w = closed_form_inverse_se3(ref_pose_w2c)

    # Compute relative poses: T_ci_cr = T_ci_w @ T_w_cr
    # This is a batched matrix multiply: (N, 4, 4) @ (4, 4) -> (N, 4, 4)
    # PyTorch broadcasts ref_pose_c2w to be multiplied by each pose in poses_w2c.
    poses_ref_to_cam = poses_w2c @ ref_pose_c2w
    
    return poses_ref_to_cam



# --- Seeding (From test_co3d.py) ---
def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# From https://github.com/facebookresearch/vggt/blob/main/vggt/utils/rotation.py
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)

# From https://github.com/facebookresearch/vggt/blob/main/vggt/utils/rotation.py
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

# From https://github.com/facebookresearch/vggt/blob/main/vggt/utils/rotation.py
def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out





# From https://github.com/facebookresearch/vggt/blob/main/vggt/utils/geometry.py
def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix
