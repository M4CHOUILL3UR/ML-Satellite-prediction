import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# 0) UTILITIES
# ============================================================

def show_df(df, name="DataFrame", n=5):
    print(f"\\n--- {name} (shape={df.shape}) ---")
    print(df.head(n).to_string(index=False))

# ============================================================
# 1) DATA LOADING
# ============================================================

def load_data(base_path):
    print(f"Loading data from {base_path}...")
    train_df = pd.read_csv(os.path.join(base_path, "jan_train.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "jan_test.csv"))
    answer_df = pd.read_csv(os.path.join(base_path, "answer_key.csv"))

    # Prepare Train Table
    # We only keep rows where we have both Real and Sim data (though usually train has both)
    train_table = train_df.copy()
    train_table["set"] = "train"

    # Prepare Test Table (Predict mode)
    # The test_df usually has '_sim' columns but NO 'x','y','z'.
    # We inject the answer key purely for final evaluation/plotting, 
    # but strictly we shouldn't use these columns for input features.
    test_preds = test_df.copy()
    
    # Inject answers for validation purposes (Ground Truth)
    target_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    test_preds[target_cols] = answer_df[target_cols].values
    test_preds["set"] = "test"

    # Concat for visualization if needed
    merged_all = pd.concat([train_table, test_preds], ignore_index=True)

    return train_table, test_preds, merged_all

# ============================================================
# 2) ORBIT CUTTER (Geometric Helper)
# ============================================================

def cut_after_one_orbit_3D(df, coord_cols=["x", "y", "z"], vel_cols=["Vx", "Vy", "Vz"]):
    """
    Cuts the trajectory after one full revolution (2pi radians) 
    based on the angular change in the orbital plane.
    """
    df = df.copy()
    
    # Check if columns exist
    if not all(c in df.columns for c in coord_cols + vel_cols):
        return df

    # Drop NaNs in trajectory
    df = df.dropna(subset=coord_cols + vel_cols)
    if len(df) < 10:
        return df

    # Extract position (r) and velocity (v)
    r = df[coord_cols].to_numpy(dtype=float)
    v = df[vel_cols].to_numpy(dtype=float)

    # Angular momentum vector h = r x v
    hv = np.cross(r, v)
    h = np.nanmean(hv, axis=0)
    hn = np.linalg.norm(h)
    
    if hn < 1e-9:
        return df # degenerate orbit

    h = h / hn

    # Define an arbitrary reference frame in the orbital plane
    # u_axis perpen to h
    u_axis = np.cross(h, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(u_axis) < 1e-3:
        u_axis = np.cross(h, np.array([0.0, 1.0, 0.0]))
    u_axis /= np.linalg.norm(u_axis)
    
    v_axis = np.cross(h, u_axis)

    # Project position onto this plane
    u = (r @ u_axis[:, None]).flatten()
    w = (r @ v_axis[:, None]).flatten()
    
    # Calculate angle theta
    theta = np.unwrap(np.arctan2(w, u))

    # Select one loop (0 to 2pi)
    if len(theta) == 0:
        return df
        
    start_theta = theta[0]
    mask = (theta - start_theta) <= 2 * np.pi
    
    return df.loc[mask].copy()

# ============================================================
# 3) FEATURE ENGINEERING
# ============================================================

def get_physics_features(df, suffix=""):
    """
    Generates physics-based features for a set of columns.
    suffix: "" for real columns (x, y...), "_sim" for simulation columns (x_sim, y_sim...)
    """
    # Map generic names to actual columns
    cx, cy, cz = f"x{suffix}", f"y{suffix}", f"z{suffix}"
    vx, vy, vz = f"Vx{suffix}", f"Vy{suffix}", f"Vz{suffix}"
    
    # Check availability
    req = [cx, cy, cz, vx, vy, vz]
    if not all(c in df.columns for c in req):
        # Return empty or partial if cols missing
        return pd.DataFrame(index=df.index)

    X = df[req].astype(float)
    
    # 1. Position Magnitude (Radius)
    r = np.sqrt(X[cx]**2 + X[cy]**2 + X[cz]**2)
    
    # 2. Velocity Magnitude (Speed)
    s = np.sqrt(X[vx]**2 + X[vy]**2 + X[vz]**2)
    
    # 3. Angular Momentum (Specific) h = r x v
    hx = X[cy]*X[vz] - X[cz]*X[vy]
    hy = X[cz]*X[vx] - X[cx]*X[vz]
    hz = X[cx]*X[vy] - X[cy]*X[vx]
    h = np.sqrt(hx**2 + hy**2 + hz**2)
    
    # 4. Energy (Specific Mechanical Energy) E = v^2/2 - mu/r
    # mu for Earth approx 398600 km^3/s^2
    mu = 398600.0
    energy = (s**2)/2 - mu/(r + 1e-9)

    feat_df = pd.DataFrame(index=df.index)
    feat_df[f"r{suffix}"] = r
    feat_df[f"speed{suffix}"] = s
    feat_df[f"h_norm{suffix}"] = h
    feat_df[f"energy{suffix}"] = energy
    
    # Cosines / Sines of position (directionality)
    feat_df[f"cos_x{suffix}"] = X[cx] / (r + 1e-9)
    feat_df[f"cos_y{suffix}"] = X[cy] / (r + 1e-9)
    feat_df[f"cos_z{suffix}"] = X[cz] / (r + 1e-9)
    
    return feat_df

def prepare_ml_data(df, target_cols=["x", "y", "z", "Vx", "Vy", "Vz"]):
    """
    Prepares X (features from SIM) and y (RESIDUAL = Real - Sim).
    """
    df = df.copy()
    
    # 1. Base Features from Simulation
    sim_cols = ["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]
    
    # Drop rows where we lack targets (Real data) or inputs (Sim data)
    # Note: For training we need both. For prediction we only need Sim.
    valid_mask = df[target_cols + sim_cols].notna().all(axis=1)
    df = df.loc[valid_mask].copy()

    # Generate derived features on SIM data
    phys_feats = get_physics_features(df, suffix="_sim")
    
    # Combine Raw Sim + Derived Sim as Input X
    X = pd.concat([df[sim_cols], phys_feats], axis=1)
    
    # Calculate Target: Residual (Real - Sim)
    y = pd.DataFrame(index=df.index)
    for col in target_cols:
        sim_col = col + "_sim"
        y[f"diff_{col}"] = df[col] - df[sim_col]
        
    return X, y

# ============================================================
# 4) MODELING
# ============================================================

def train_correction_model(train_df):
    """
    Trains a model to predict the error of the simulation.
    """
    print("\\n[ML] Preparing training data...")
    X, y = prepare_ml_data(train_df)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[ML] Training shape: {X_train.shape}")
    
    # Using RandomForest. 
    # Note: HistGradientBoostingRegressor is often better for this but RF is standard.
    model = RandomForestRegressor(
        n_estimators=50,      # kept low for speed given 'student' project
        max_depth=15,         # limit depth to prevent overfitting
        n_jobs=-1,
        random_state=42
    )
    
    
    
    print("[ML] Fitting Random Forest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"[ML] Validation RMSE (on residuals): {rmse:.4f}")
    
    return model, X.columns.tolist(), y.columns.tolist()

def apply_correction(df, model, feature_cols, target_diff_cols):
    """
    Applies the model to predict 'Real' values given 'Sim' values.
    Real_pred = Sim + Model(Sim)
    """
    # Prepare Features (Sim only)
    # We use the same logic as training but we don't need 'Real' columns here
    sim_cols = ["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]
    
    # Ensure no NaNs in sim inputs
    mask = df[sim_cols].notna().all(axis=1)
    df_valid = df.loc[mask].copy()
    
    if len(df_valid) == 0:
        return df 
        
    phys_feats = get_physics_features(df_valid, suffix="_sim")
    X = pd.concat([df_valid[sim_cols], phys_feats], axis=1)
    
    # Align columns
    X = X[feature_cols]
    
    # Predict Residuals
    diff_preds = model.predict(X)
    
    # Add residuals to Sim columns to get Corrected predictions
    out_df = df.copy()
    
    # Map back: diff_x -> x, etc.
    # target_diff_cols are like ['diff_x', 'diff_y', ...]
    base_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    
    for i, col in enumerate(base_cols):
        # We write to a new column 'x_pred', 'y_pred'...
        # prediction = sim + predicted_diff
        out_df.loc[mask, f"{col}_pred"] = df_valid[f"{col}_sim"] + diff_preds[:, i]
        
    return out_df

# ============================================================
# 5) VISUALIZATION
# ============================================================

def plot_matplotlib_3d_sample(df, nb_sats=15, R_earth=6371):
    """
    Plots a sample of satellites in 3D using Matplotlib (static plot).
    """
    valid_sats = df["sat_id"].unique()
    if len(valid_sats) == 0:
        return

    sample_sats = np.random.choice(valid_sats, size=min(nb_sats, len(valid_sats)), replace=False)
    print(f"\\n[Viz] Plotting sample of {len(sample_sats)} satellites (Matplotlib)...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe Earth
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightblue", alpha=0.3, linewidth=0)

    # Colors
    colors = plt.cm.jet(np.linspace(0, 1, len(sample_sats)))

    for i, sat in enumerate(sample_sats):
        sub = df[df["sat_id"] == sat].copy()
        # Clean numeric
        sub = sub.dropna(subset=["x", "y", "z", "Vx", "Vy", "Vz"])
        if len(sub) == 0: continue
        
        orbit = cut_after_one_orbit_3D(sub, ["x", "y", "z"], ["Vx", "Vy", "Vz"])
        
        if len(orbit) > 0:
            ax.plot(orbit.x, orbit.y, orbit.z, color=colors[i], label=f"Sat {sat}")

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("3D Orbits Sample")
    
    # Check if too many legends
    if len(sample_sats) <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        
    plt.tight_layout()
    plt.show()

def compare_trajectories(df, sat_id, R_earth=6371):
    """
    Plots Real vs Sim vs Corrected for a single satellite (Plotly).
    """
    sub = df[df["sat_id"] == sat_id].copy()
    
    if len(sub) < 10:
        print(f"Not enough data for Sat {sat_id}")
        return

    # Cut 1 orbit for cleaner viz
    # Real
    real_orbit = cut_after_one_orbit_3D(sub, ["x", "y", "z"], ["Vx", "Vy", "Vz"])
    # Sim
    sim_orbit = cut_after_one_orbit_3D(sub, ["x_sim", "y_sim", "z_sim"], ["Vx_sim", "Vy_sim", "Vz_sim"])
    # Pred
    pred_orbit = cut_after_one_orbit_3D(sub, ["x_pred", "y_pred", "z_pred"], ["Vx_pred", "Vy_pred", "Vz_pred"])

    fig = go.Figure()
    
    # Earth
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
    fig.add_surface(x=xs, y=ys, z=zs, colorscale="Blues", opacity=0.6, showscale=False)

    # Traces
    fig.add_trace(go.Scatter3d(x=real_orbit.x, y=real_orbit.y, z=real_orbit.z,
                               mode='lines', name='Real (Ground Truth)', line=dict(color='cyan', width=5)))
    
    fig.add_trace(go.Scatter3d(x=sim_orbit.x_sim, y=sim_orbit.y_sim, z=sim_orbit.z_sim,
                               mode='lines', name='Simulation (SGP4)', line=dict(color='orange', width=3, dash='dash')))
                               
    fig.add_trace(go.Scatter3d(x=pred_orbit.x_pred, y=pred_orbit.y_pred, z=pred_orbit.z_pred,
                               mode='lines', name='ML Corrected', line=dict(color='lime', width=4)))

    fig.update_layout(
        title=f"Trajectory Correction: Sat {sat_id}",
        template="plotly_dark",
        scene=dict(aspectmode='data')
    )
    fig.show()

# ============================================================
# MAIN
# ============================================================

def main():
    # 1. Setup
    path = os.getcwd()
    
    # 2. Load
    train_table, test_preds, merged_all = load_data(path)
    
    # 3. Viz 1: Overview
    # Plotting this BEFORE training to show "what we have"
    plot_matplotlib_3d_sample(merged_all)
    
    # 4. Train Model (Sim -> Residual)
    # We train on a subset to be fast (or full set if time permits)
    # For this demo, let's use a sample of 20 satellites to speed up
    sample_sats = train_table["sat_id"].unique()[:20] 
    train_subset = train_table[train_table["sat_id"].isin(sample_sats)]
    
    model, feats, targets = train_correction_model(train_subset)
    
    # 5. Apply to Test Set (or a holdout validation satellite)
    print("\\n[Prediction] Applying model to Test set...")
    test_res = apply_correction(test_preds, model, feats, targets)
    
    # 6. Evaluation & Viz 2: Detailed Correction
    # Let's pick a random satellite from the Test set to visualize
    test_sats = test_res["sat_id"].unique()
    if len(test_sats) > 0:
        sid = np.random.choice(test_sats)
        print(f"\\nVisualizing Satellite {sid} from Test Set...")
        compare_trajectories(test_res, sid)
        
        # Calculate improvement metric on this satellite
        sub = test_res[test_res["sat_id"] == sid]
        base_err = np.linalg.norm(sub[["x","y","z"]].values - sub[["x_sim","y_sim","z_sim"]].values, axis=1).mean()
        ml_err = np.linalg.norm(sub[["x","y","z"]].values - sub[["x_pred","y_pred","z_pred"]].values, axis=1).mean()
        
        print(f"--- Sat {sid} Error Analysis ---")
        print(f"Base Simulation Error: {base_err:.3f} km")
        print(f"ML Corrected Error:    {ml_err:.3f} km")
        print(f"Improvement:           {(base_err - ml_err):.3f} km")
        
    print("\\nDone.")

if __name__ == "__main__":
    main()
