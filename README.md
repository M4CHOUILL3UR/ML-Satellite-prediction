# Satellite Trajectory Correction using Random Forests

This project applies Machine Learning to correct errors in standard SGP4 satellite tracking simulations.

The goal is to predict the difference (residuals) between the SGP4 simulation and the actual ground truth data provided in the dataset. By correcting these errors, we can generate a more accurate trajectory than the simulation alone.

## Method

### The Problem with Cartesian Coordinates
Initial tests trying to predict errors directly on `x, y, z` coordinates failed. Since satellites move at high velocities (~7 km/s), small timing errors resulted in large spatial errors. The model struggled to learn the physics, resulting in jagged, discontinuous orbits that looked like noise.

### The Keplerian Approach
To fix this, I converted the simulation state into **Keplerian Orbital Elements**:
* **Semi-major axis (a) & Eccentricity (e):** define the shape of the ellipse.
* **Inclination (i):** defines the tilt.
* **Angular elements (Ω, ω, M):** define the orientation and position.

By training the Random Forest on these elements instead of raw coordinates, the model learns to correct the *geometry* of the orbit rather than just point locations.

**Feature Engineering Note:**
All angular features were decomposed into `sin()` and `cos()` components. This prevents the model from seeing a discontinuity where 0 degrees meets 360 degrees, which was causing instability in earlier versions.

## Setup

Requires Python 3.8+ and the following packages:

```bash
pip install numpy pandas matplotlib plotly scikit-learn