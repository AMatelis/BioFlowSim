import numpy as np
import matplotlib.pyplot as plt

try:
    from ipywidgets import interact
except ImportError:
    interact = None


def plot_trajectories(positions, step=1, save_path=None):
    """
    Overlay particle trajectories over the entire simulation.

    Args:
        positions (np.ndarray): shape (frames, num_particles, 2)
        step (int): skip frames for clarity
        save_path (str, optional): path to save figure
    """
    frames, num_particles, _ = positions.shape

    plt.figure(figsize=(6, 6))

    for p in range(num_particles):
        traj = positions[::step, p, :]
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1)
        plt.scatter(traj[-1, 0], traj[-1, 1], s=20, c="red")

    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.gca().invert_yaxis()
    plt.title("Particle Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_density_heatmap(positions, frame_idx=0, bins=50, save_path=None):
    """
    Plot particle density heatmap for a given frame.

    Args:
        positions (np.ndarray): shape (frames, num_particles, 2)
        frame_idx (int): frame index to visualize
        bins (int): number of bins for 2D histogram
        save_path (str, optional): path to save figure
    """
    frame_pos = positions[frame_idx]

    heatmap, _, _ = np.histogram2d(
        frame_pos[:, 0],
        frame_pos[:, 1],
        bins=bins,
        range=[[0, 256], [0, 256]]
    )

    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="hot",
        interpolation="nearest"
    )
    plt.colorbar(label="Particle Density")
    plt.title(f"Density Heatmap - Frame {frame_idx}")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()


def plot_velocity_heatmap(positions, frame_idx=1, bins=50, save_path=None):
    """
    Plot velocity magnitude heatmap using difference between frames.

    Args:
        positions (np.ndarray): shape (frames, num_particles, 2)
        frame_idx (int): frame index to visualize
        bins (int): number of bins for 2D histogram
        save_path (str, optional): path to save figure
    """
    if frame_idx <= 0:
        raise ValueError("frame_idx must be greater than 0")

    velocity = positions[frame_idx] - positions[frame_idx - 1]
    speed = np.linalg.norm(velocity, axis=1)
    frame_pos = positions[frame_idx]

    heatmap, _, _ = np.histogram2d(
        frame_pos[:, 0],
        frame_pos[:, 1],
        bins=bins,
        weights=speed,
        range=[[0, 256], [0, 256]]
    )

    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="viridis",
        interpolation="nearest"
    )
    plt.colorbar(label="Speed")
    plt.title(f"Velocity Heatmap - Frame {frame_idx}")

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()


def interactive_visualization(positions, particle_size=20, color="red"):
    """
    Interactive visualization of particle positions per frame.

    Args:
        positions (np.ndarray): shape (frames, num_particles, 2)
        particle_size (int): marker size
        color (str): marker color
    """
    if interact is None:
        raise ImportError(
            "ipywidgets is required for interactive visualization. "
            "Install it with `pip install ipywidgets`."
        )

    frames = positions.shape[0]

    @interact(frame_idx=(0, frames - 1))
    def show_frame(frame_idx=0):
        plt.figure(figsize=(6, 6))
        plt.scatter(
            positions[frame_idx, :, 0],
            positions[frame_idx, :, 1],
            s=particle_size,
            c=color
        )
        plt.xlim(0, 256)
        plt.ylim(0, 256)
        plt.gca().invert_yaxis()
        plt.title(f"Frame {frame_idx}")
        plt.show()


def overlay_trajectories_on_video(
    frames,
    positions,
    step=1,
    output_path="overlay.avi",
    fps=30,
    particle_radius=2
):
    """
    Overlay particle trajectories on video frames and save as video.

    Args:
        frames (np.ndarray): shape (num_frames, height, width)
        positions (np.ndarray): shape (num_frames, num_particles, 2)
        step (int): skip frames for trajectories
        output_path (str): output video file
        fps (int): frames per second
        particle_radius (int): radius of particles
    """
    import cv2

    num_frames, height, width = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=False
    )

    for i in range(num_frames):
        frame = frames[i].astype(np.float32)

        for p in range(positions.shape[1]):
            x, y = positions[i, p]

            if i > 0 and i % step == 0:
                x_prev, y_prev = positions[i - step, p]
                cv2.line(
                    frame,
                    (int(x_prev), int(y_prev)),
                    (int(x), int(y)),
                    255,
                    1
                )

            cv2.circle(
                frame,
                (int(x), int(y)),
                particle_radius,
                255,
                -1
            )

        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(frame_uint8)

    writer.release()
    print(f"Overlay video saved to {output_path}")