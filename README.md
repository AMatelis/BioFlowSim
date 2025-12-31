# BioFlowSim

BioFlowSim is a **Python library for simulating 2D particle flow in microfluidic channels**, generating **realistic imaging sequences**, and creating **ML-ready datasets**. Designed for biomedical engineering students, researchers, and AI practitioners working on flow imaging, speckle imaging, or microfluidic experiments.

---

## Features

- **2D Particle/Cell Flow Simulation**
  - User-defined velocity, diffusion, particle density
  - Stable simulation with boundary reflection and obstacle support
- **Realistic Imaging Artifacts**
  - Gaussian and Poisson noise
  - Speckle patterns for laser-based flow imaging
  - Motion blur and lighting variations
- **Video and Frame Export**
  - `.avi`, `.mp4` video sequences
  - Optional per-frame `.png` images for ML pipelines
- **PyTorch Dataset Integration**
  - Generates ML-ready inputs and labels
  - Supports sequence sampling, batch loading, and optional transforms
- **Visualization & Analysis**
  - Particle trajectories overlay
  - Density and velocity heatmaps
  - Interactive debug mode with sliders
- **Extensible**
  - Designed modularly for branching channels, 3D flows, and advanced imaging

---

## Installation

```bash
git clone https://github.com/yourusername/BioFlowSim.git
cd BioFlowSim
pip install -e .
