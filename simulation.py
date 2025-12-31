import numpy as np


class FlowSimulator:
    """
    Simulates 2D particle flow in microfluidic channels with obstacles and
    non-uniform velocity fields.
    """

    def __init__(self, num_particles=100, frames=100, width=256, height=256,
                 base_velocity=(1.0, 0.0), diffusion=0.1, particle_sizes=None,
                 obstacles=None, velocity_field=None, random_seed=42):
        """
        Initialize the simulation.

        Parameters:
        ----------
        num_particles : int
            Number of particles.
        frames : int
            Number of simulation frames.
        width, height : int
            Simulation domain dimensions.
        base_velocity : tuple(float, float)
            Default velocity if velocity_field is not provided.
        diffusion : float
            Brownian motion intensity.
        particle_sizes : array-like
            Radii for particles (optional, default = 1 pixel).
        obstacles : list of dict
            Obstacles in channel: {'type': 'circle'/'rect', 'params': {...}}
        velocity_field : callable or np.ndarray
            If callable: f(x, y) -> (vx, vy). If array: shape (height, width, 2)
        random_seed : int
            Seed for reproducibility.
        """
        self.num_particles = num_particles
        self.frames = frames
        self.width = width
        self.height = height
        self.base_velocity = np.array(base_velocity, dtype=np.float32)
        self.diffusion = diffusion
        self.particle_sizes = np.ones(num_particles) if particle_sizes is None else particle_sizes
        self.obstacles = obstacles if obstacles else []
        self.velocity_field = velocity_field
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Initialize particle positions randomly
        self.positions = self.rng.uniform([0,0], [self.width, self.height], size=(num_particles, 2))
        # Store trajectories, velocities, and collisions
        self.trajectories = np.zeros((frames, num_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((frames, num_particles, 2), dtype=np.float32)
        self.collisions = np.zeros((frames, num_particles), dtype=bool)

    def _get_velocity(self, pos):
        """
        Compute velocity for each particle at given positions.
        Supports uniform base velocity or user-defined velocity field.
        """
        if callable(self.velocity_field):
            return np.array([self.velocity_field(x, y) for x, y in pos], dtype=np.float32)
        elif isinstance(self.velocity_field, np.ndarray):
            x_idx = np.clip(pos[:,0].astype(int), 0, self.width-1)
            y_idx = np.clip(pos[:,1].astype(int), 0, self.height-1)
            return self.velocity_field[y_idx, x_idx]
        else:
            return np.tile(self.base_velocity, (self.num_particles,1))

    def _apply_boundaries(self, pos):
        """
        Reflect particles at channel boundaries.
        """
        pos[:,0] = np.clip(pos[:,0], 0, self.width-1)
        pos[:,1] = np.clip(pos[:,1], 0, self.height-1)
        return pos

    def _check_obstacles(self, pos):
        """
        Simple obstacle reflection or marking collision.
        """
        collision_flags = np.zeros(self.num_particles, dtype=bool)
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                cx, cy, r = obs['params']['cx'], obs['params']['cy'], obs['params']['r']
                dist = np.linalg.norm(pos - np.array([cx, cy]), axis=1)
                mask = dist < r
                pos[mask] += (r - dist[mask])[:,None]  # simple push out
                collision_flags |= mask
            # Rectangle or other shapes can be added here
        return pos, collision_flags

    def step(self, frame_idx):
        """
        Advance simulation by one frame.
        """
        vel = self._get_velocity(self.positions)
        noise = self.rng.normal(0, self.diffusion, size=(self.num_particles,2))
        new_pos = self.positions + vel + noise

        new_pos = self._apply_boundaries(new_pos)
        new_pos, collision_flags = self._check_obstacles(new_pos)

        self.velocities[frame_idx] = new_pos - self.positions
        self.collisions[frame_idx] = collision_flags
        self.positions = new_pos
        self.trajectories[frame_idx] = self.positions.copy()

    def run(self):
        """
        Run simulation for all frames.

        Returns:
        -------
        trajectories : np.ndarray
            Particle positions per frame (frames, num_particles, 2)
        velocities : np.ndarray
            Per-frame particle velocities
        collisions : np.ndarray
            Boolean array of collisions per frame
        """
        for f in range(self.frames):
            self.step(f)
        return self.trajectories, self.velocities, self.collisions


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    def parabolic_velocity(x, y):
        # Example: laminar parabolic flow in y-direction
        width = 256
        vx = 0
        vy = 0.01 * (y - width/2)**2
        return np.array([vx, vy], dtype=np.float32)

    obstacles = [{'type':'circle','params':{'cx':128,'cy':128,'r':20}}]

    sim = FlowSimulator(num_particles=50, frames=100, velocity=(0.5,0.2),
                        diffusion=0.05, obstacles=obstacles,
                        velocity_field=None)
    traj, vel, col = sim.run()

    print("Trajectories shape:", traj.shape)
    print("Velocities shape:", vel.shape)
    print("Collision flags shape:", col.shape)