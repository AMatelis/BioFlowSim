"""Tests for BioFlowSim's imaging utilities.

Run from the repository root:
    python -m pytest test_imaging.py -v
"""

import numpy as np
import pytest
from bioflowsim import imaging


def test_render_frame_returns_grayscale_uint8_image():
    positions = np.array([[10, 12], [20, 25]], dtype=float)

    frame = imaging.render_frame(
        positions, width=64, height=48, particle_radius=2
    )

    assert frame.shape == (48, 64)
    assert frame.dtype == np.uint8
    assert frame.max() > 0
    assert frame.min() == 0


def test_render_frame_supports_three_channels():
    positions = np.array([[10, 12]], dtype=float)

    frame = imaging.render_frame(
        positions, width=32, height=24, multi_channel=True
    )

    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.uint8
    assert frame.max() == 255


def test_render_frame_uses_per_particle_intensity():
    positions = np.array([[8, 8], [24, 24]], dtype=float)
    intensities = np.array([0.25, 1.0], dtype=float)

    frame = imaging.render_frame(
        positions,
        width=40,
        height=40,
        particle_radius=0,
        particle_intensity=intensities,
    )

    assert frame[8, 8] in (63, 64)
    assert frame[24, 24] == 255


def test_render_frame_clips_positions_to_image_bounds():
    positions = np.array([[-10, -5], [100, 100]], dtype=float)

    frame = imaging.render_frame(
        positions, width=20, height=15, particle_radius=0
    )

    assert frame[0, 0] == 255
    assert frame[14, 19] == 255


def test_gaussian_noise_with_zero_sigma_leaves_frame_unchanged():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)

    result = imaging.add_gaussian_noise(frame, sigma=0)

    np.testing.assert_array_equal(result, frame)
    assert result.dtype == np.uint8


def test_gaussian_noise_preserves_shape_and_pixel_range():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(32, 24), dtype=np.uint8)

    result = imaging.add_gaussian_noise(frame, sigma=12)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_poisson_noise_preserves_shape_and_pixel_range():
    frame = np.full((16, 20), 100, dtype=np.uint8)

    result = imaging.add_poisson_noise(frame)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_poisson_noise_handles_black_frame():
    frame = np.zeros((8, 8), dtype=np.uint8)

    result = imaging.add_poisson_noise(frame)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_zero_intensity_speckle_leaves_frame_unchanged():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)

    result = imaging.add_speckle(frame, intensity=0)

    np.testing.assert_array_equal(result, frame)


def test_speckle_preserves_shape_dtype_and_pixel_range():
    frame = np.full((24, 24), 180, dtype=np.uint8)

    result = imaging.add_speckle(frame, intensity=0.2)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_motion_blur_preserves_shape_and_dtype():
    frame = np.zeros((32, 32), dtype=np.uint8)
    frame[16, 8:24] = 255

    result = imaging.add_motion_blur(frame, ksize=5, angle=0)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_adjust_lighting_factor_one_leaves_frame_unchanged():
    frame = np.array([[0, 50, 100, 255]], dtype=np.uint8)

    result = imaging.adjust_lighting(frame, factor=1.0)

    np.testing.assert_array_equal(result, frame)


def test_adjust_lighting_clips_values_to_uint8_range():
    frame = np.array([[100, 200]], dtype=np.uint8)

    result = imaging.adjust_lighting(frame, factor=2.0)

    expected = np.array([[200, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8


def test_generate_video_writes_frames_and_saves_pngs(tmp_path, monkeypatch):
    class FakeVideoWriter:
        def __init__(self):
            self.frames = []
            self.released = False

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    fake_writer = FakeVideoWriter()

    monkeypatch.setattr(imaging.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(
        imaging.cv2, "VideoWriter", lambda *args, **kwargs: fake_writer
    )

    trajectories = np.array(
        [
            [[5.0, 5.0], [10.0, 10.0]],
            [[6.0, 5.0], [11.0, 10.0]],
            [[7.0, 5.0], [12.0, 10.0]],
        ],
        dtype=np.float32,
    )

    frame_dir = tmp_path / "frames"
    output_path = tmp_path / "test.avi"

    imaging.generate_video(
        trajectories,
        output_path=str(output_path),
        width=32,
        height=24,
        save_frames=True,
        frame_dir=str(frame_dir),
    )

    assert len(fake_writer.frames) == trajectories.shape[0]
    assert fake_writer.released
    assert all(frame.shape == (24, 32) for frame in fake_writer.frames)

    assert sorted(path.name for path in frame_dir.glob("*.png")) == [
        "frame_0000.png",
        "frame_0001.png",
        "frame_0002.png",
    ]