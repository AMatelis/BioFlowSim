import numpy as np
import cv2
import os


# ============================================================
# Frame Rendering
# ============================================================

def render_frame(
    positions,
    width=256,
    height=256,
    particle_radius=2,
    particle_intensity=None,
    multi_channel=False
):
    """
    Render particle positions into an image frame.

    Parameters
    ----------
    positions : np.ndarray
        Shape (num_particles, 2), particle x,y coordinates
    width, height : int
        Frame dimensions
    particle_radius : int
        Particle radius in pixels
    particle_intensity : float or np.ndarray
        Scalar or per particle intensity in range [0, 1]
    multi_channel : bool
        If True, returns a 3 channel image

    Returns
    -------
    frame : np.ndarray
        uint8 image of shape (H, W) or (H, W, 3)
    """

    if multi_channel:
        frame = np.zeros((height, width, 3), dtype=np.float32)
    else:
        frame = np.zeros((height, width), dtype=np.float32)

    if particle_intensity is None:
        particle_intensity = np.ones(len(positions), dtype=np.float32)

    for i, (x, y) in enumerate(positions):
        xi = int(np.clip(x, 0, width - 1))
        yi = int(np.clip(y, 0, height - 1))
        intensity = float(particle_intensity[i])

        if multi_channel:
            cv2.circle(
                frame,
                (xi, yi),
                particle_radius,
                (intensity, intensity, intensity),
                -1
            )
        else:
            cv2.circle(
                frame,
                (xi, yi),
                particle_radius,
                intensity,
                -1
            )

    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame


# ============================================================
# Imaging Artifacts
# ============================================================

def add_gaussian_noise(frame, sigma=5.0):
    """
    Add additive Gaussian noise.
    """
    noise = np.random.normal(0.0, sigma, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_poisson_noise(frame):
    """
    Add Poisson photon counting noise.
    """
    frame_f = frame.astype(np.float32)
    scale = 2 ** np.ceil(np.log2(np.max(frame_f) + 1))
    noisy = np.random.poisson(frame_f * scale) / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_speckle(frame, intensity=0.1):
    """
    Add multiplicative speckle noise.
    """
    noise = np.random.uniform(
        1.0 - intensity,
        1.0 + intensity,
        frame.shape
    )
    speckled = frame.astype(np.float32) * noise
    return np.clip(speckled, 0, 255).astype(np.uint8)


def add_motion_blur(frame, ksize=5, angle=0.0):
    """
    Apply directional motion blur.
    """
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    rad = np.deg2rad(angle)

    x = int(center + np.cos(rad) * center)
    y = int(center + np.sin(rad) * center)

    cv2.line(kernel, (center, center), (x, y), 1.0, thickness=1)
    kernel_sum = kernel.sum()
    if kernel_sum != 0:
        kernel /= kernel_sum

    return cv2.filter2D(frame, -1, kernel)


def adjust_lighting(frame, factor=1.0):
    """
    Apply global brightness scaling.
    """
    adjusted = frame.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ============================================================
# Video Generation
# ============================================================

def generate_video(
    trajectories,
    output_path="flow.avi",
    fps=30,
    width=256,
    height=256,
    particle_radius=2,
    noise_type=None,
    noise_level=5.0,
    speckle_intensity=0.0,
    motion_blur=False,
    blur_ksize=5,
    lighting_var=False,
    multi_channel=False,
    save_frames=False,
    frame_dir="frames"
):
    """
    Generate a video from particle trajectories with imaging artifacts.

    Parameters
    ----------
    trajectories : np.ndarray
        Shape (frames, num_particles, 2)
    """

    num_frames = trajectories.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
        isColor=multi_channel
    )

    if save_frames:
        os.makedirs(frame_dir, exist_ok=True)

    for i in range(num_frames):
        frame = render_frame(
            trajectories[i],
            width=width,
            height=height,
            particle_radius=particle_radius,
            multi_channel=multi_channel
        )

        if noise_type == "gaussian":
            frame = add_gaussian_noise(frame, sigma=noise_level)
        elif noise_type == "poisson":
            frame = add_poisson_noise(frame)

        if speckle_intensity > 0.0:
            frame = add_speckle(frame, intensity=speckle_intensity)

        if motion_blur:
            angle = np.random.uniform(0.0, 360.0)
            frame = add_motion_blur(frame, ksize=blur_ksize, angle=angle)

        if lighting_var:
            factor = np.random.uniform(0.8, 1.2)
            frame = adjust_lighting(frame, factor)

        if save_frames:
            frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)

        if multi_channel and frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        writer.write(frame)

    writer.release()
    print(f"[BioFlowSim] Video saved to {output_path}")


# ============================================================
# Example Execution
# ============================================================

if __name__ == "__main__":
    from simulation import FlowSimulator

    sim = FlowSimulator(
        num_particles=50,
        frames=100,
        velocity=(0.5, 0.2),
        diffusion=0.05
    )

    trajectories, _, _ = sim.run()

    generate_video(
        trajectories,
        output_path="demo_imaging.avi",
        fps=30,
        particle_radius=2,
        noise_type="gaussian",
        noise_level=10,
        speckle_intensity=0.05,
        motion_blur=True,
        blur_ksize=5,
        lighting_var=True,
        multi_channel=False,
        save_frames=True,
        frame_dir="demo_frames"
    )