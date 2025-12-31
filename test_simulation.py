
import numpy as np
from bioflowsim.simulation import FlowSimulator

def test_stability():
    """
    Test that particles never leave the defined channel boundaries.
    """
    sim = FlowSimulator(num_particles=50, frames=100, width=256, height=256)
    positions = sim.run()
    
    assert np.all(positions[:,:,0] >= 0) and np.all(positions[:,:,0] <= sim.width-1), "X positions out of bounds"
    assert np.all(positions[:,:,1] >= 0) and np.all(positions[:,:,1] <= sim.height-1), "Y positions out of bounds"
    print("Stability test passed: all particles inside channel boundaries.")

def test_reproducibility():
    """
    Test that two simulations with the same random seed produce identical trajectories.
    """
    seed = 123
    sim1 = FlowSimulator(num_particles=30, frames=50, random_seed=seed)
    sim2 = FlowSimulator(num_particles=30, frames=50, random_seed=seed)
    
    traj1 = sim1.run()
    traj2 = sim2.run()
    
    assert np.allclose(traj1, traj2), "Simulations with the same seed do not match"
    print("Reproducibility test passed: identical trajectories with same seed.")

def test_basic_functionality():
    """
    Test that simulation returns correct shape and values.
    """
    num_particles = 20
    frames = 10
    sim = FlowSimulator(num_particles=num_particles, frames=frames, velocity=(1.0, 0.5), diffusion=0.1)
    traj = sim.run()
    
    assert traj.shape == (frames, num_particles, 2), f"Trajectory shape mismatch: {traj.shape}"
    assert np.all(traj >= 0), "Trajectory has negative coordinates"
    print("Basic functionality test passed: trajectory shape and values are valid.")

if __name__ == "__main__":
    test_stability()
    test_reproducibility()
    test_basic_functionality()
    print("All tests passed successfully.")