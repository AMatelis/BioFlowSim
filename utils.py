import numpy as np
import os

def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy and optionally other libraries.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def validate_positive_int(value, name="parameter"):
    """
    Validate that a value is a positive integer.
    
    Args:
        value: value to check
        name: name of the parameter for error message
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")

def validate_positive_float(value, name="parameter"):
    """
    Validate that a value is a positive float.
    
    Args:
        value: value to check
        name: name of the parameter for error message
    """
    if not isinstance(value, (float, int)) or value <= 0:
        raise ValueError(f"{name} must be a positive float, got {value}.")

def ensure_dir(path):
    """
    Create directory if it does not exist.
    
    Args:
        path (str): directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_numpy_array(array, path):
    """
    Save numpy array to file (.npy format).
    
    Args:
        array (np.ndarray): array to save
        path (str): output file path
    """
    np.save(path, array)
    print(f"Array saved to {path}")

def load_numpy_array(path):
    """
    Load numpy array from .npy file.
    
    Args:
        path (str): file path
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")
    return np.load(path)

def normalize_frame(frame):
    """
    Normalize frame to [0,1] float32.
    
    Args:
        frame (np.ndarray): input frame
    """
    frame = frame.astype(np.float32)
    frame = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-8)
    return frame

def compute_velocity(positions):
    """
    Compute per-frame velocities for particles.
    
    Args:
        positions (np.ndarray): shape (frames, num_particles, 2)
    
    Returns:
        velocities (np.ndarray): shape (frames-1, num_particles, 2)
    """
    return positions[1:] - positions[:-1]

def positions_to_density_map(positions, height=256, width=256, bins=50):
    """
    Convert particle positions to a 2D density map for ML labels or visualization.
    
    Args:
        positions (np.ndarray): shape (num_particles, 2)
        height (int): frame height
        width (int): frame width
        bins (int): number of bins along each axis
    
    Returns:
        density_map (np.ndarray): 2D histogram normalized to [0,1]
    """
    heatmap, _, _ = np.histogram2d(positions[:,0], positions[:,1],
                                   bins=bins, range=[[0, width],[0, height]])
    heatmap = heatmap.astype(np.float32)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap