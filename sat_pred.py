import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D plotting
import plotly.graph_objects as go
import plotly.io as pio 

# Force Plotly to open in browser
pio.renderers.default = "browser"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# 0) UTILITIES & MATH HELPERS
# ============================================================

def show_df(df, name="DataFrame", n=5):
    print(f"\n--- {name} (shape={df.shape}) ---")
    print(df.head(n).to_string(index=False))

def angle_diff(a, b):
    """
    Calculates smallest difference between two angles in radians.
    Solves the 'wrapping' problem: diff(0.1, 6.2) should be small, not big.
    """
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

# ============================================================
# 1) DATA LOADING
# ============================================================

def load_data(base_path):
    print(f"Loading data from {base_path}...")
    train_df = pd.read_csv(os.path.join(base_path, "jan_train.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "jan_test.csv"))
    answer_df = pd.read_csv(os.path.join(base_path, "answer_key.csv"))

    # Prepare Train Table
    train_table = train_df.copy()
    train_table["set"] = "train"

    # Prepare Test Table
    test_preds = test_df.copy()
    
    # Inject answers for validation purposes (Ground Truth)
    target_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    test_preds[target_cols] = answer_df[target_cols].values
    test_preds["set"] = "test"

    return train_table, test_preds

# ============================================================
# 2) KEPLERIAN MATH
# ============================================================

MU_EARTH = 398600.4418  # km^3/s^2

def cart2kep(r_vec, v_vec):
    """
    Vectorized Cartesian -> Keplerian conversion.
    Returns: [a, e, i, Omega, w, M]
    """
    # Magnitudes
    r = np.linalg.norm(r_vec, axis=1)
    v = np.linalg.norm(v_vec, axis=1)
    
    # Angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=1)
    
    # Node vector
    n_x = -h_vec[:, 1]
    n_y = h_vec[:, 0]
    n = np.sqrt(n_x**2 + n_y**2)
    
    # Eccentricity
    r_dot_v = np.sum(r_vec * v_vec, axis=1)
    term1 = (v**2 - MU_EARTH/r)[:, np.newaxis] * r_vec
    term2 = r_dot_v[:, np.newaxis] * v_vec
    e_vec = (term1 - term2) / MU_EARTH
    e = np.linalg.norm(e_vec, axis=1)
    
    # Semi-major axis
    specific_energy = (v**2)/2 - MU_EARTH/r
    a = -MU_EARTH / (2 * specific_energy)
    
    # Inclination
    inc = np.arccos(np.clip(h_vec[:, 2] / h, -1, 1))
    
    # RAAN (Omega)
    Omega = np.arccos(np.clip(n_x / (n + 1e-9), -1, 1))
    Omega[n_y < 0] = 2*np.pi - Omega[n_y < 0]
    
    # Argument of Perigee (w)
    n_dot_e = n_x*e_vec[:,0] + n_y*e_vec[:,1]
    w = np.arccos(np.clip(n_dot_e / (n * e + 1e-9), -1, 1))
    w[e_vec[:, 2] < 0] = 2*np.pi - w[e_vec[:, 2] < 0]
    
    # True Anomaly (nu)
    e_dot_r = np.sum(e_vec * r_vec, axis=1)
    nu = np.arccos(np.clip(e_dot_r / (e * r + 1e-9), -1, 1))
    nu[r_dot_v < 0] = 2*np.pi - nu[r_dot_v < 0]
    
    # Mean Anomaly (M)
    E_ecc = 2 * np.arctan(np.sqrt((1 - e)/(1 + e + 1e-9)) * np.tan(nu/2))
    M = E_ecc - e * np.sin(E_ecc)
    
    return np.column_stack([a, e, inc, Omega, w, M])

def kep2cart(kep_arr):
    """
    Vectorized Keplerian -> Cartesian conversion.
    Input: [a, e, i, Omega, w, M]
    """
    a, e, inc, Omega, w, M = kep_arr.T
    
    # Solve Kepler Eq for E (approx)
    E = M.copy()
    for _ in range(5):
        E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        
    # True Anomaly
    sqrt_term = np.sqrt((1+e)/(1-e + 1e-9))
    nu = 2 * np.arctan(sqrt_term * np.tan(E/2))
    
    # Perifocal coords
    r_dist = a * (1 - e * np.cos(E))
    p = r_dist * np.cos(nu)
    q = r_dist * np.sin(nu)
    
    h = np.sqrt(MU_EARTH * a * (1 - e**2 + 1e-9))
    mu_over_h = MU_EARTH / (h + 1e-9)
    vp = -mu_over_h * np.sin(nu)
    vq =  mu_over_h * (e + np.cos(nu))
    
    # Rotation to ECI
    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(w), np.sin(w)
    ci, si = np.cos(inc), np.sin(inc)
    
    P_x = cO*cw - sO*sw*ci
    P_y = sO*cw + cO*sw*ci
    P_z = sw*si
    
    Q_x = -cO*sw - sO*cw*ci
    Q_y = -sO*sw + cO*cw*ci
    Q_z = cw*si
    
    x = p*P_x + q*Q_x
    y = p*P_y + q*Q_y
    z = p*P_z + q*Q_z
    
    vx = vp*P_x + vq*Q_x
    vy = vp*P_y + vq*Q_y
    vz = vp*P_z + vq*Q_z
    
    return np.column_stack([x, y, z, vx, vy, vz])

# ============================================================
# 3) FEATURE ENGINEERING (SMART ANGLES)
# ============================================================

def get_keplerian_data(df, is_train=True):
    # 1. Convert SIM data to Keplerian
    r_sim = df[["x_sim", "y_sim", "z_sim"]].values
    v_sim = df[["Vx_sim", "Vy_sim", "Vz_sim"]].values
    k_sim = cart2kep(r_sim, v_sim) # [a, e, i, Om, w, M]
    
    # 2. Build Feature Matrix X (Transform angles to sin/cos!)
    # This prevents the model from seeing a "jump" at 360 degrees
    X = pd.DataFrame(index=df.index)
    X["a_sim"] = k_sim[:, 0]
    X["e_sim"] = k_sim[:, 1]
    
    # For all angles (idx 2,3,4,5), add sin and cos components
    angle_names = ["i", "Om", "w", "M"]
    for idx, name in enumerate(angle_names, start=2):
        X[f"sin_{name}"] = np.sin(k_sim[:, idx])
        X[f"cos_{name}"] = np.cos(k_sim[:, idx])
        
    # Extra physics feature: Mean Motion n
    X["n_motion"] = np.sqrt(MU_EARTH / X["a_sim"]**3)

    y = None
    if is_train:
        # 3. Compute Targets (Errors)
        r_true = df[["x", "y", "z"]].values
        v_true = df[["Vx", "Vy", "Vz"]].values
        k_true = cart2kep(r_true, v_true)
        
        y = pd.DataFrame(index=df.index)
        
        # Simple difference for a and e
        y["d_a"] = k_true[:, 0] - k_sim[:, 0]
        y["d_e"] = k_true[:, 1] - k_sim[:, 1]
        
        # Angular difference for others (handling wrap)
        for idx, name in enumerate(angle_names, start=2):
            y[f"d_{name}"] = angle_diff(k_true[:, idx], k_sim[:, idx])
            
    return X, y, k_sim

# ============================================================
# 4) MODELING
# ============================================================

def train_correction_model(train_df):
    print("\n[ML] Engineering Keplerian Features...")
    
    # Filter valid data
    train_df = train_df.dropna(subset=["x_sim", "x"])
    X, y, _ = get_keplerian_data(train_df, is_train=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest with Smoothing
    # min_samples_leaf=5 prevents the model from being too "twitchy"
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,  # <--- Key for smoothing
        n_jobs=-1,
        random_state=42
    )
    
    print("[ML] Training Random Forest...")
    model.fit(X_train, y_train)
    
    print(f"[ML] Validation Score (R^2): {model.score(X_val, y_val):.4f}")
    return model, X.columns.tolist()

def apply_correction(df, model, feature_cols):
    print("Applying corrections...")
    valid_mask = df[["x_sim", "y_sim", "z_sim"]].notna().all(axis=1)
    df_valid = df.loc[valid_mask].copy()
    
    if len(df_valid) == 0: return df
    
    # Prepare Inputs
    X, _, k_sim = get_keplerian_data(df_valid, is_train=False)
    X = X[feature_cols]
    
    # Predict Angular Errors
    pred_diffs = model.predict(X)
    
    # Apply Corrections
    # k_sim structure: [a, e, i, Om, w, M]
    # pred structure:  [d_a, d_e, d_i, d_Om, d_w, d_M]
    
    k_corr = k_sim.copy()
    k_corr[:, 0] += pred_diffs[:, 0] # a
    k_corr[:, 1] += pred_diffs[:, 1] # e
    
    # For angles, just add the diff (wrap handled by sin/cos in reconstruction)
    for i in range(4):
        k_corr[:, i+2] += pred_diffs[:, i+2]
        
    # Convert back to Cartesian
    cart_corr = kep2cart(k_corr)
    
    out_df = df.copy()
    coords = ["x", "y", "z", "Vx", "Vy", "Vz"]
    for i, col in enumerate(coords):
        out_df.loc[valid_mask, f"{col}_pred"] = cart_corr[:, i]
        
    return out_df

# ============================================================
# 5) VISUALIZATION
# ============================================================

def cut_one_orbit(df, suffix=""):
    """Simple geometric cut for plotting 1 orbit"""
    if len(df) < 100: return df
    # Just take first 600 points (approx 1 orbit for LEO) to keep it simple
    return df.iloc[:600]

def visualize_multiple(df, sat_ids):
    fig = go.Figure()
    
    # Earth Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = 6371 * np.cos(u)*np.sin(v)
    y = 6371 * np.sin(u)*np.sin(v)
    z = 6371 * np.cos(v)
    fig.add_surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.3, showscale=False)

    colors = ['cyan', 'magenta', 'lime', 'yellow']
    
    for i, sid in enumerate(sat_ids):
        sub = df[df["sat_id"] == sid].sort_values("id") # Ensure time order
        if len(sub) == 0: continue
        
        # Cut to first orbit for clarity
        sub = cut_one_orbit(sub)
        
        c = colors[i % len(colors)]
        
        # Ground Truth (Solid)
        fig.add_trace(go.Scatter3d(
            x=sub.x, y=sub.y, z=sub.z,
            mode='lines', name=f'Sat {sid} (True)',
            line=dict(color=c, width=4)
        ))
        
        # ML Prediction (Dashed)
        fig.add_trace(go.Scatter3d(
            x=sub.x_pred, y=sub.y_pred, z=sub.z_pred,
            mode='lines', name=f'Sat {sid} (ML)',
            line=dict(color='white', width=3, dash='dash')
        ))

    fig.update_layout(
        title="Multi-Satellite Trajectory Correction",
        template="plotly_dark",
        scene=dict(aspectmode='data')
    )
    fig.show()

# ============================================================
# MAIN
# ============================================================

def main():
    path = os.getcwd()
    train_table, test_preds = load_data(path)
    
    # Train on a subset for speed
    train_subset = train_table[train_table["sat_id"] < 200] 
    
    model, features = train_correction_model(train_subset)
    
    print("\n[Prediction] Applying model to Test set...")
    results = apply_correction(test_preds, model, features)
    
    # Pick 3 random satellites to visualize
    sample_sats = np.random.choice(results["sat_id"].unique(), 3, replace=False)
    print(f"\nVisualizing Satellites: {sample_sats}")
    
    visualize_multiple(results, sample_sats)

if __name__ == "__main__":
    main()