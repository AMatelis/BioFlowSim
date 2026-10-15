

import numpy as np
from bioflowsim.simulation import FlowSimulator


def test_stability():
    """Test that particles remain within channel boundaries."""
    sim = FlowSimulator(
        num_particles=50,
        frames=100,
        width=256,
        height=256,
    )

    trajectories, velocities, collisions = sim.run()

    assert np.all(trajectories[:, :, 0] >= 0), "Negative X positions"
    assert np.all(trajectories[:, :, 0] <= sim.width - 1), "X positions out of bounds"
    assert np.all(trajectories[:, :, 1] >= 0), "Negative Y positions"
    assert np.all(trajectories[:, :, 1] <= sim.height - 1), "Y positions out of bounds"


def test_reproducibility():
    """Test that identical seeds produce identical simulation results."""
    seed = 123

    sim1 = FlowSimulator(
        num_particles=30,
        frames=50,
        random_seed=seed,
    )
    sim2 = FlowSimulator(
        num_particles=30,
        frames=50,
        random_seed=seed,
    )

    traj1, vel1, col1 = sim1.run()
    traj2, vel2, col2 = sim2.run()

    assert np.allclose(traj1, traj2), "Trajectories differ"
    assert np.allclose(vel1, vel2), "Velocities differ"
    assert np.array_equal(col1, col2), "Collision results differ"


def test_basic_functionality():
    """Test simulation output shapes and values."""
    num_particles = 20
    frames = 10

    sim = FlowSimulator(
        num_particles=num_particles,
        frames=frames,
        base_velocity=(1.0, 0.5),
        diffusion=0.1,
    )

    trajectories, velocities, collisions = sim.run()

    assert trajectories.shape == (frames, num_particles, 2)
    assert velocities.shape == (frames, num_particles, 2)
    assert collisions.shape == (frames, num_particles)

    assert np.all(np.isfinite(trajectories)), "Trajectories contain invalid values"
    assert np.all(np.isfinite(velocities)), "Velocities contain invalid values"
    assert np.all(trajectories >= 0), "Trajectories contain negative coordinates"
    assert np.all(trajectories[:, :, 0] <= sim.width - 1), "X positions out of bounds"
    assert np.all(trajectories[:, :, 1] <= sim.height - 1), "Y positions out of bounds"
