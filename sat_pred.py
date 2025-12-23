import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import HistGradientBoostingRegressor # <--- FASTEST sklearn model
from sklearn.preprocessing import StandardScaler

# Force Plotly to open in browser
pio.renderers.default = "browser"

# ============================================================
# 1) ORBITAL MECHANICS: THE RIC FRAME (The "Clockwork" Math)
# ============================================================
MU_EARTH = 398600.4418

def eci2ric(r_sim, v_sim, r_true, v_true):
    """
    Converts errors from Earth-Centered Inertial (ECI) to 
    Radial-Intrack-Crosstrack (RIC) frame.
    """
    r_norm = np.linalg.norm(r_sim, axis=1)
    uR = r_sim / r_norm[:, None]
    
    h_vec = np.cross(r_sim, v_sim)
    h_norm = np.linalg.norm(h_vec, axis=1)
    uC = h_vec / h_norm[:, None]
    
    uI = np.cross(uC, uR)
    
    delta_r = r_true - r_sim
    
    # Project Error onto RIC Vectors
    d_rad = np.sum(delta_r * uR, axis=1)
    d_int = np.sum(delta_r * uI, axis=1)
    d_cro = np.sum(delta_r * uC, axis=1)
    
    return np.column_stack([d_rad, d_int, d_cro])

def ric2eci(r_sim, v_sim, ric_corrections):
    """
    Converts predicted RIC corrections back to ECI to reconstruct the path.
    """
    r_norm = np.linalg.norm(r_sim, axis=1)
    uR = r_sim / r_norm[:, None]
    
    h_vec = np.cross(r_sim, v_sim)
    h_norm = np.linalg.norm(h_vec, axis=1)
    uC = h_vec / h_norm[:, None]
    
    uI = np.cross(uC, uR)
    
    d_rad = ric_corrections[:, 0][:, None]
    d_int = ric_corrections[:, 1][:, None]
    d_cro = ric_corrections[:, 2][:, None]
    
    delta_r = (d_rad * uR) + (d_int * uI) + (d_cro * uC)
    
    return r_sim + delta_r

# ============================================================
# 2) FEATURE ENGINEERING (Standard Keplerian Input)
# ============================================================

def cart2kep_simple(r_vec, v_vec):
    """Simplified conversion for feature generation."""
    r = np.linalg.norm(r_vec, axis=1)
    v = np.linalg.norm(v_vec, axis=1)
    h = np.linalg.norm(np.cross(r_vec, v_vec), axis=1)
    
    energy = (v**2)/2 - MU_EARTH/r
    a = -MU_EARTH / (2 * energy)
    
    e_vec = np.cross(v_vec, np.cross(r_vec, v_vec))/MU_EARTH - r_vec/r[:,None]
    e = np.linalg.norm(e_vec, axis=1)
    
    inc = np.arccos(np.clip(np.cross(r_vec, v_vec)[:, 2] / h, -1, 1))
    
    return np.column_stack([a, e, inc])

def get_ric_data(df, is_train=True):
    r_sim = df[["x_sim", "y_sim", "z_sim"]].values
    v_sim = df[["Vx_sim", "Vy_sim", "Vz_sim"]].values
    
    kep_sim = cart2kep_simple(r_sim, v_sim) # [a, e, i]
    
    X = pd.DataFrame(index=df.index)
    X["a_sim"] = kep_sim[:, 0]
    X["e_sim"] = kep_sim[:, 1]
    X["i_sim"] = kep_sim[:, 2]
    
    # Time proxy (normalized index per satellite)
    X["time_step"] = df.groupby("sat_id").cumcount()
    
    y = None
    if is_train:
        r_true = df[["x", "y", "z"]].values
        v_true = df[["Vx", "Vy", "Vz"]].values
        
        # Target: The RIC Error
        ric_errors = eci2ric(r_sim, v_sim, r_true, v_true)
        y = pd.DataFrame(ric_errors, columns=["dR", "dI", "dC"], index=df.index)
        
    return X, y

# ============================================================
# 3) MODELING (SPEED OPTIMIZED)
# ============================================================

def train_ric_model(train_df):
    print("[Mission Control] Converting to RIC Frame & Training...")
    train_df = train_df.dropna(subset=["x_sim", "x"])
    
    X, y = get_ric_data(train_df, is_train=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # HistGradientBoosting is usually 10-50x faster than standard GradientBoosting
    model = HistGradientBoostingRegressor(
        max_iter=100,          # Reduced iterations
        max_depth=5,           # Shallow trees for speed
        learning_rate=0.1,
        random_state=42
    )
    
    from sklearn.multioutput import MultiOutputRegressor
    multi_model = MultiOutputRegressor(model, n_jobs=-1) # Parallelized
    
    print("[Mission Control] Fitting Fast RIC Model...")
    multi_model.fit(X_scaled, y)
    
    return multi_model, scaler, X.columns.tolist()

def apply_ric_correction(df, model, scaler, feature_cols):
    valid_mask = df[["x_sim", "y_sim", "z_sim"]].notna().all(axis=1)
    df_valid = df.loc[valid_mask].copy()
    
    X, _ = get_ric_data(df_valid, is_train=False)
    X_scaled = scaler.transform(X)
    
    preds_ric = model.predict(X_scaled)
    
    r_sim = df_valid[["x_sim", "y_sim", "z_sim"]].values
    v_sim = df_valid[["Vx_sim", "Vy_sim", "Vz_sim"]].values
    
    r_corrected = ric2eci(r_sim, v_sim, preds_ric)
    
    out_df = df.copy()
    for i, col in enumerate(["x", "y", "z"]):
        out_df.loc[valid_mask, f"{col}_pred"] = r_corrected[:, i]
        
    return out_df

# ============================================================
# 4) VISUALIZATION
# ============================================================

def visualize_ric_mission(df, sat_id, exag_factor=20.0):
    sub = df[df["sat_id"] == sat_id].sort_values("id")
    if len(sub) == 0: return
    
    # Plot first orbit only for clarity
    sub = sub.iloc[:1000] 
    
    print(f"[Viz] Sat {sat_id} (RIC Correction, {exag_factor}x Exag)")
    
    fig = go.Figure()
    
    r_sim = sub[["x_sim", "y_sim", "z_sim"]].values
    r_true = sub[["x", "y", "z"]].values
    r_ml = sub[["x_pred", "y_pred", "z_pred"]].values
    
    r_true_plot = r_sim + (r_true - r_sim) * exag_factor
    r_ml_plot = r_sim + (r_ml - r_sim) * exag_factor
    
    # 1. Baseline Sim (Orange)
    fig.add_trace(go.Scatter3d(
        x=r_sim[:,0], y=r_sim[:,1], z=r_sim[:,2],
        mode='lines', name='SGP4 Baseline',
        line=dict(color='orange', width=2, dash='dashdot'), opacity=0.4
    ))
    
    # 2. Ground Truth (Cyan)
    fig.add_trace(go.Scatter3d(
        x=r_true_plot[:,0], y=r_true_plot[:,1], z=r_true_plot[:,2],
        mode='lines', name=f'Ground Truth ({exag_factor}x)',
        line=dict(color='cyan', width=6)
    ))
    
    # 3. RIC Corrected (Lime Green)
    fig.add_trace(go.Scatter3d(
        x=r_ml_plot[:,0], y=r_ml_plot[:,1], z=r_ml_plot[:,2],
        mode='lines', name=f'RIC Corrected ({exag_factor}x)',
        line=dict(color='lime', width=4)
    ))
    
    # Earth
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = 6371 * np.cos(u)*np.sin(v)
    y = 6371 * np.sin(u)*np.sin(v)
    z = 6371 * np.cos(v)
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.1, showscale=False))

    fig.update_layout(title=f"Sat {sat_id}: RIC Correction", template="plotly_dark", scene=dict(aspectmode='data'))
    fig.show()

# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    train_full = pd.read_csv("jan_train.csv")
    
    # SPEED FILTER: Only use 25 good satellites for training
    counts = train_full["sat_id"].value_counts()
    valid_sats = counts[counts > 500].index.tolist()
    
    train_sats = valid_sats[:25] # Reduced from 40 to 25
    demo_sats = valid_sats[25:30]
    
    train_data = train_full[train_full["sat_id"].isin(train_sats)].copy()
    demo_data = train_full[train_full["sat_id"].isin(demo_sats)].copy()
    
    # Train
    model, scaler, cols = train_ric_model(train_data)
    
    # Apply
    print("Applying corrections...")
    res_list = []
    for sid in demo_sats:
        sub = demo_data[demo_data["sat_id"] == sid].copy().sort_values("id")
        res_list.append(apply_ric_correction(sub, model, scaler, cols))
    
    demo_res = pd.concat(res_list)
    
    # Visualize first result
    visualize_ric_mission(demo_res, demo_sats[0], exag_factor=30.0)

if __name__ == "__main__":
    main()