from .simulation import FlowSimulator

# Imaging
from .imaging import (
    render_frame,
    add_gaussian_noise,
    add_poisson_noise,
    add_speckle,
    add_motion_blur,
    adjust_lighting,
    generate_video
)

# Dataset
from .dataset import FlowVideoDataset, create_dataloader, compute_velocity, positions_to_density_map

# Visualization
from .visualization import (
    plot_trajectories,
    plot_density_heatmap,
    plot_velocity_heatmap,
    interactive_visualization
)

# Utilities
from .utils import *

# Define what is accessible when importing *
__all__ = [
    "FlowSimulator",
    "render_frame",
    "add_gaussian_noise",
    "add_poisson_noise",
    "add_speckle",
    "add_motion_blur",
    "adjust_lighting",
    "generate_video",
    "FlowVideoDataset",
    "create_dataloader",
    "compute_velocity",
    "positions_to_density_map",
    "plot_trajectories",
    "plot_density_heatmap",
    "plot_velocity_heatmap",
    "interactive_visualization",
    "utils"
]