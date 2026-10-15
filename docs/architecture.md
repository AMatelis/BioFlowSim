# Architecture

## Overview

BioFlowSim is organized around the generation of synthetic particle-flow data and its conversion into imaging outputs.

## Main Components

### Simulation

The simulation module manages particle movement and the generation of trajectories under configured flow conditions.

### Imaging

The imaging module converts simulated particle positions into image data and can support imaging artifacts, depending on the implemented functions.

### Dataset

The dataset module organizes generated data for analysis and potential machine-learning workflows.

### Visualization

The visualization module provides tools for inspecting simulated motion and generated outputs.

### Utilities

Utility functions support common operations used across the project.

## Data Flow

The intended workflow is:

1. Define the simulation configuration.
2. Generate particle trajectories.
3. Convert trajectories into synthetic images.
4. Apply supported imaging effects.
5. Visualize or export the results.
6. Prepare the generated data for downstream analysis.

## Extensibility

Potential extensions include additional flow geometries, more realistic imaging models, expanded dataset formats, and improved validation against experimental measurements.

These extensions should be considered future development unless they are already implemented.