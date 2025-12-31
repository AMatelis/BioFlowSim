import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class FlowVideoDataset(Dataset):
    """
    PyTorch Dataset for simulated particle flow videos.
    
    Each sample can be a single frame or a sequence of frames,
    with labels corresponding to particle positions, velocities, or flow maps.
    """
    def __init__(self, frames, positions, sequence_length=1, transform=None, normalize=True):
        """
        Parameters:
        ----------
        frames : np.ndarray
            Shape: (num_frames, height, width) or (num_frames, height, width, channels)
        positions : np.ndarray
            Shape: (num_frames, num_particles, 2)
        sequence_length : int
            Number of consecutive frames per sample
        transform : callable
            Optional torchvision or custom transform applied to frame sequence
        normalize : bool
            If True, scale frame pixel values to [0,1]
        """
        assert frames.shape[0] == positions.shape[0], "Frames and positions must have same number of frames"
        self.frames = frames.astype(np.float32)
        if normalize:
            self.frames /= 255.0
        self.positions = positions.astype(np.float32)
        self.sequence_length = sequence_length
        self.transform = transform

        self.num_frames = frames.shape[0]
        self.samples = self.num_frames - sequence_length + 1

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # Sequence of frames
        frame_seq = self.frames[idx:idx+self.sequence_length]
        # Convert to PyTorch tensor: (seq_len, C, H, W)
        if frame_seq.ndim == 3:
            frame_seq = torch.tensor(frame_seq).unsqueeze(1)  # add channel dimension
        else:  # already multi-channel
            frame_seq = torch.tensor(frame_seq).permute(0, 3, 1, 2)

        # Labels: particle positions for last frame in sequence
        label = torch.tensor(self.positions[idx+self.sequence_length-1], dtype=torch.float32)

        # Apply transforms if provided
        if self.transform:
            frame_seq = self.transform(frame_seq)

        return frame_seq, label


# -----------------------------
# DataLoader Utility
# -----------------------------
def create_dataloader(frames, positions, batch_size=8, shuffle=True,
                      sequence_length=1, transform=None, normalize=True, num_workers=0):
    """
    Create a PyTorch DataLoader from frames and particle positions.

    Parameters:
    ----------
    frames : np.ndarray
        Video frames (num_frames, H, W) or (num_frames, H, W, C)
    positions : np.ndarray
        Particle positions (num_frames, num_particles, 2)
    batch_size : int
    shuffle : bool
    sequence_length : int
    transform : callable
    normalize : bool
        Normalize pixel values to [0,1]
    num_workers : int
        For multi-processing DataLoader
    """
    dataset = FlowVideoDataset(frames, positions, sequence_length, transform, normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# -----------------------------
# Optional Helper Functions
# -----------------------------
def compute_velocity(positions):
    """
    Compute per-frame particle velocities from positions
    Shape: (frames-1, num_particles, 2)
    """
    return positions[1:] - positions[:-1]


def positions_to_density_map(positions, height=256, width=256, bins=50):
    """
    Convert particle positions to 2D density maps per frame
    Returns: np.array of shape (num_frames, bins, bins)
    """
    num_frames = positions.shape[0]
    density_maps = np.zeros((num_frames, bins, bins), dtype=np.float32)
    for f in range(num_frames):
        hist, _, _ = np.histogram2d(
            positions[f, :, 0], positions[f, :, 1],
            bins=bins, range=[[0, width], [0, height]]
        )
        density_maps[f] = hist
    return density_maps


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    from simulation import FlowSimulator

    # Run simulation
    sim = FlowSimulator(num_particles=50, frames=100, velocity=(0.5, 0.2), diffusion=0.05)
    traj = sim.run()

    # Generate frames for dataset (simple white particles on black)
    frames = np.zeros((traj.shape[0], 256, 256), dtype=np.uint8)
    for i, pos in enumerate(traj):
        for x, y in pos:
            frames[i, int(y), int(x)] = 255

    # Create DataLoader
    dataloader = create_dataloader(frames, traj, batch_size=8, sequence_length=5)

    # Iterate over batches
    for batch_frames, batch_labels in dataloader:
        print("Batch frames shape:", batch_frames.shape)  # (batch_size, seq_len, 1, H, W)
        print("Batch labels shape:", batch_labels.shape)  # (batch_size, num_particles, 2)
        break