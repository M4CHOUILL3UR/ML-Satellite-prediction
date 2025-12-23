import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio 

# Force Plotly to open in browser (fixes Spyder "invisible plot" issue)
pio.renderers.default = "browser"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# 0) UTILITIES
# ============================================================

def show_df(df, name="DataFrame", n=5):
    print(f"\n--- {name} (shape={df.shape}) ---")
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
    train_table = train_df.copy()
    train_table["set"] = "train"

    # Prepare Test Table
    test_preds = test_df.copy()
    
    # Inject answers for validation purposes (Ground Truth)
    target_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    test_preds[target_cols] = answer_df[target_cols].values
    test_preds["set"] = "test"

    # Concat for visualization
    merged_all = pd.concat([train_table, test_preds], ignore_index=True)

    return train_table, test_preds, merged_all

# ============================================================
# 2) ORBIT CUTTER (Geometric Helper)
# ============================================================

def cut_after_one_orbit_3D(df, coord_cols=["x", "y", "z"], vel_cols=["Vx", "Vy", "Vz"]):
    """
    Cuts the trajectory after one full revolution (2pi radians).
    """
    df = df.copy()
    
    if not all(c in df.columns for c in coord_cols + vel_cols):
        return df

    df = df.dropna(subset=coord_cols + vel_cols)
    if len(df) < 10:
        return df

    r = df[coord_cols].to_numpy(dtype=float)
    v = df[vel_cols].to_numpy(dtype=float)

    hv = np.cross(r, v)
    h = np.nanmean(hv, axis=0)
    hn = np.linalg.norm(h)
    
    if hn < 1e-9:
        return df 

    h = h / hn
    u_axis = np.cross(h, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(u_axis) < 1e-3:
        u_axis = np.cross(h, np.array([0.0, 1.0, 0.0]))
    u_axis /= np.linalg.norm(u_axis)
    
    v_axis = np.cross(h, u_axis)

    u = (r @ u_axis[:, None]).flatten()
    w = (r @ v_axis[:, None]).flatten()
    
    theta = np.unwrap(np.arctan2(w, u))

    if len(theta) == 0:
        return df
        
    start_theta = theta[0]
    mask = (theta - start_theta) <= 2 * np.pi
    
    return df.loc[mask].copy()

# ============================================================
# 3) KEPLERIAN MATH (The hard part...)
# ============================================================

MU_EARTH = 398600.4418  # Standard gravitational parameter for Earth (km^3/s^2)

def cart2kep(r_vec, v_vec):
    """
    Convert Cartesian state (r, v) to Keplerian elements (a, e, i, Omega, omega, M).
    Using standard orbital mechanics formulas. 
    """
    # Magnitudes
    r = np.linalg.norm(r_vec, axis=1)
    v = np.linalg.norm(v_vec, axis=1)
    
    # Angular momentum h = r x v
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=1)
    
    # Node vector n = k x h (k is [0,0,1])
    # k x h = [-hy, hx, 0]
    n_x = -h_vec[:, 1]
    n_y = h_vec[:, 0]
    n = np.sqrt(n_x**2 + n_y**2)
    
    # Eccentricity vector
    # e = (1/mu) * [ (v^2 - mu/r)*r - (r.v)*v ]
    r_dot_v = np.sum(r_vec * v_vec, axis=1)
    
    term1 = (v**2 - MU_EARTH/r)[:, np.newaxis] * r_vec
    term2 = r_dot_v[:, np.newaxis] * v_vec
    e_vec = (term1 - term2) / MU_EARTH
    e = np.linalg.norm(e_vec, axis=1)
    
    # Energy & Semi-major axis (a)
    # E = v^2/2 - mu/r = -mu / 2a
    specific_energy = (v**2)/2 - MU_EARTH/r
    # careful with parabolic/hyperbolic cases, but sats should be elliptic
    a = -MU_EARTH / (2 * specific_energy)
    
    # Inclination i = acos(hz / h)
    inc = np.arccos(np.clip(h_vec[:, 2] / h, -1, 1))
    
    # Right Ascension of Ascending Node (Omega)
    # Omega = acos(nx / n)
    Omega = np.arccos(np.clip(n_x / (n + 1e-9), -1, 1))
    # If ny < 0, Omega = 2pi - Omega
    mask_ny = n_y < 0
    Omega[mask_ny] = 2*np.pi - Omega[mask_ny]
    
    # Argument of Perigee (w)
    # w = acos( (n . e) / (n * e) )
    n_dot_e = n_x*e_vec[:,0] + n_y*e_vec[:,1] # nz is 0
    w = np.arccos(np.clip(n_dot_e / (n * e + 1e-9), -1, 1))
    # if ez < 0, w = 2pi - w
    mask_ez = e_vec[:, 2] < 0
    w[mask_ez] = 2*np.pi - w[mask_ez]
    
    # True Anomaly (nu)
    # nu = acos( (e . r) / (e * r) )
    e_dot_r = np.sum(e_vec * r_vec, axis=1)
    nu = np.arccos(np.clip(e_dot_r / (e * r + 1e-9), -1, 1))
    # if r.v < 0, nu = 2pi - nu
    mask_rv = r_dot_v < 0
    nu[mask_rv] = 2*np.pi - nu[mask_rv]
    
    # Mean Anomaly (M) - ML likes this better than nu because it changes linearly with time
    # tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
    E_ecc = 2 * np.arctan(np.sqrt((1 - e)/(1 + e + 1e-9)) * np.tan(nu/2))
    # M = E - e sin E
    M = E_ecc - e * np.sin(E_ecc)
    
    # Stack 'em
    return np.column_stack([a, e, inc, Omega, w, M])

def kep2cart(kep_arr):
    """
    Go back from Keplerian (a, e, i, Omega, w, M) to Cartesian (x, y, z, Vx, Vy, Vz).
    We need this to turn our ML predictions back into the submission format.
    """
    a = kep_arr[:, 0]
    e = kep_arr[:, 1]
    inc = kep_arr[:, 2]
    Omega = kep_arr[:, 3]
    w = kep_arr[:, 4]
    M = kep_arr[:, 5]
    
    # 1. Solve Kepler's Equation for Eccentric Anomaly (E)
    # Iterative method (Newton-Raphson) because M = E - e sin E can't be inverted algebraically
    # Start guess E = M
    E = M.copy()
    for _ in range(5): # 5 iterations usually enough for small e
        f = E - e*np.sin(E) - M
        f_prime = 1 - e*np.cos(E)
        E = E - f / f_prime
        
    # 2. Get True Anomaly (nu)
    # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    sqrt_term = np.sqrt((1+e)/(1-e + 1e-9))
    nu = 2 * np.arctan(sqrt_term * np.tan(E/2))
    
    # 3. Distance r
    r_dist = a * (1 - e * np.cos(E))
    
    # 4. Position/Velocity in Perifocal Frame (PQW)
    # p = x, q = y (in the orbital plane)
    p = r_dist * np.cos(nu)
    q = r_dist * np.sin(nu)
    
    # velocity components
    # h = sqrt(mu * a * (1-e^2))
    h = np.sqrt(MU_EARTH * a * (1 - e**2 + 1e-9))
    mu_over_h = MU_EARTH / (h + 1e-9)
    
    vp = -mu_over_h * np.sin(nu)
    vq =  mu_over_h * (e + np.cos(nu))
    
    # 5. Rotate to Geocentric Equatorial Frame (IJK)
    # Using rotation matrices for Omega, inc, w
    # It's a mess of sin/cos, so I'll write it out explicitly
    
    cO = np.cos(Omega)
    sO = np.sin(Omega)
    cw = np.cos(w)
    sw = np.sin(w)
    ci = np.cos(inc)
    si = np.sin(inc)
    
    # Unit vectors
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
# 4) FEATURE ENGINEERING & DATA PREP
# ============================================================

def get_keplerian_diffs(df):
    """
    This function drives the logic:
    1. Take SIMULATION Cartesian -> SIM Keplerian
    2. Take TRUE Cartesian -> TRUE Keplerian
    3. Calculate difference (True - Sim) to use as ML target
    """
    
    # Grab arrays
    r_sim = df[["x_sim", "y_sim", "z_sim"]].values
    v_sim = df[["Vx_sim", "Vy_sim", "Vz_sim"]].values
    
    # We only have truth for training data
    if "x" in df.columns:
        r_true = df[["x", "y", "z"]].values
        v_true = df[["Vx", "Vy", "Vz"]].values
        
        # Convert both
        k_sim = cart2kep(r_sim, v_sim)
        k_true = cart2kep(r_true, v_true)
        
        # Calculate errors: diff = True - Sim
        # (Target for the ML model)
        diffs = k_true - k_sim
        
        # Handle angle wrap-around for angular elements if needed?
        # Ideally yes, but for small corrections standard diff is usually ok.
        
        return k_sim, diffs
    else:
        # For test set (prediction), we only have sim
        k_sim = cart2kep(r_sim, v_sim)
        return k_sim, None

def prepare_ml_data(df, is_train=True):
    df = df.copy()
    
    # Filter NaNs
    req_cols = ["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]
    if is_train:
        req_cols += ["x", "y", "z", "Vx", "Vy", "Vz"]
        
    mask = df[req_cols].notna().all(axis=1)
    df = df.loc[mask].copy()

    # Get Keplerian elements
    # k_sim has 6 cols: [a, e, i, Omega, w, M]
    k_sim, k_diffs = get_keplerian_diffs(df)
    
    # Create Feature Matrix X
    # We feed the model the Simulation Keplerian elements
    # because that's what we know at test time.
    X = pd.DataFrame(k_sim, columns=["a_sim", "e_sim", "i_sim", "Om_sim", "w_sim", "M_sim"], index=df.index)
    
    # Maybe add some extra features to help the trees?
    X["n_rev"] = np.sqrt(MU_EARTH / X["a_sim"]**3) # Mean motion
    
    if is_train:
        # Target y: The ERROR in those elements
        y = pd.DataFrame(k_diffs, columns=["d_a", "d_e", "d_i", "d_Om", "d_w", "d_M"], index=df.index)
        return X, y, df
    else:
        return X, None, df

# ============================================================
# 5) MODELING
# ============================================================

def train_correction_model(train_df):
    print("\n[ML] Converting to Keplerian & Preparing training data...")
    X, y, _ = prepare_ml_data(train_df, is_train=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[ML] Training shape: {X_train.shape}")
    
    # Using Random Forest
    # predicting small corrections to orbital elements
    model = RandomForestRegressor(
        n_estimators=100,      # bumped this up a bit
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
    
    print("[ML] Fitting Random Forest to Orbital Elements...")
    model.fit(X_train, y_train)
    
    y_pred_val = model.predict(X_val)
    # Just printing RMSE of the first element (Semi-major axis) as a check
    rmse_a = np.sqrt(mean_squared_error(y_val["d_a"], y_pred_val[:, 0]))
    print(f"[ML] Validation RMSE (Semi-major axis 'a'): {rmse_a:.4f} km")
    
    return model, X.columns.tolist(), y.columns.tolist()

def apply_correction(df, model, feature_cols, target_cols):
    print("Applying corrections via Keplerian transformation...")
    
    # Prepare features
    X, _, df_valid = prepare_ml_data(df, is_train=False)
    
    if len(df_valid) == 0:
        return df 
        
    X = X[feature_cols] # Ensure column order
    
    # 1. Predict the errors (delta Keplerian)
    pred_diffs = model.predict(X)
    
    # 2. Get the Simulation Keplerian elements again
    # (We already computed them in prepare_ml_data, but let's grab from X for clarity)
    # Note: X has columns ["a_sim", "e_sim", "i_sim", "Om_sim", "w_sim", "M_sim", "n_rev"]
    # We need the first 6
    k_sim = X.iloc[:, :6].values
    
    # 3. Corrected Keplerian = Sim + Predicted_Diff
    k_corrected = k_sim + pred_diffs
    
    # 4. Convert Corrected Keplerian -> Cartesian (x, y, z...)
    # This gives us our final ML predictions in the format we need
    cart_pred = kep2cart(k_corrected)
    
    # 5. Store back in dataframe
    out_df = df.copy()
    base_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    
    # Align indices using df_valid
    for i, col in enumerate(base_cols):
        # We need to map the numpy array back to the original indices
        out_df.loc[df_valid.index, f"{col}_pred"] = cart_pred[:, i]
        
    return out_df

# ============================================================
# 6) VISUALIZATION (Kept mostly same)
# ============================================================

def plot_matplotlib_3d_sample(df, nb_sats=15, R_earth=6371):
    valid_sats = df["sat_id"].unique()
    if len(valid_sats) == 0: return

    sample_sats = np.random.choice(valid_sats, size=min(nb_sats, len(valid_sats)), replace=False)
    print(f"\n[Viz] Plotting sample of {len(sample_sats)} satellites (Matplotlib)...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightblue", alpha=0.3, linewidth=0)

    colors = plt.cm.jet(np.linspace(0, 1, len(sample_sats)))

    for i, sat in enumerate(sample_sats):
        sub = df[df["sat_id"] == sat].copy()
        sub = sub.dropna(subset=["x", "y", "z", "Vx", "Vy", "Vz"])
        if len(sub) == 0: continue
        
        orbit = cut_after_one_orbit_3D(sub, ["x", "y", "z"], ["Vx", "Vy", "Vz"])
        
        if len(orbit) > 0:
            ax.plot(orbit.x, orbit.y, orbit.z, color=colors[i], label=f"Sat {sat}")

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("3D Orbits Sample")
    
    if len(sample_sats) <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        
    plt.tight_layout()
    plt.show()

def compare_trajectories(df, sat_id, R_earth=6371):
    sub = df[df["sat_id"] == sat_id].copy()
    
    if len(sub) < 10:
        print(f"Not enough data for Sat {sat_id}")
        return

    # Use a try-except here because sometimes the cutter fails on bad predictions
    try:
        real_orbit = cut_after_one_orbit_3D(sub, ["x", "y", "z"], ["Vx", "Vy", "Vz"])
        sim_orbit = cut_after_one_orbit_3D(sub, ["x_sim", "y_sim", "z_sim"], ["Vx_sim", "Vy_sim", "Vz_sim"])
        pred_orbit = cut_after_one_orbit_3D(sub, ["x_pred", "y_pred", "z_pred"], ["Vx_pred", "Vy_pred", "Vz_pred"])
    except Exception as e:
        print(f"Viz Error: {e}")
        return

    fig = go.Figure()
    
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
    fig.add_surface(x=xs, y=ys, z=zs, colorscale="Blues", opacity=0.6, showscale=False)

    fig.add_trace(go.Scatter3d(x=real_orbit.x, y=real_orbit.y, z=real_orbit.z,
                               mode='lines', name='Real (Ground Truth)', line=dict(color='cyan', width=5)))
    
    fig.add_trace(go.Scatter3d(x=sim_orbit.x_sim, y=sim_orbit.y_sim, z=sim_orbit.z_sim,
                               mode='lines', name='Simulation', line=dict(color='orange', width=3, dash='dash')))
                               
    fig.add_trace(go.Scatter3d(x=pred_orbit.x_pred, y=pred_orbit.y_pred, z=pred_orbit.z_pred,
                               mode='lines', name='ML Corrected (Keplerian)', line=dict(color='lime', width=4)))

    fig.update_layout(
        title=f"Trajectory Correction (Kepler Space): Sat {sat_id}",
        template="plotly_dark",
        scene=dict(aspectmode='data')
    )
    print("Opening Plotly figure in browser...")
    fig.show()

# ============================================================
# MAIN
# ============================================================

def main():
    path = os.getcwd()
    
    # 1. Load
    train_table, test_preds, merged_all = load_data(path)
    
    # 2. Viz 1: Overview
    # plot_matplotlib_3d_sample(merged_all)
    
    # 3. Train
    # Using subset for speed in testing
    sample_sats = train_table["sat_id"].unique()[:20] 
    train_subset = train_table[train_table["sat_id"].isin(sample_sats)]
    
    model, feats, targets = train_correction_model(train_subset)
    
    # 4. Apply
    print("\n[Prediction] Applying model to Test set...")
    test_res = apply_correction(test_preds, model, feats, targets)
    
    # 5. Viz 2: Detailed
    test_sats = test_res["sat_id"].unique()
    if len(test_sats) > 0:
        sid = np.random.choice(test_sats)
        print(f"\nVisualizing Satellite {sid} from Test Set...")
        compare_trajectories(test_res, sid)
        
        sub = test_res[test_res["sat_id"] == sid]
        if "x" in sub.columns and "x_pred" in sub.columns:
            base_err = np.linalg.norm(sub[["x","y","z"]].values - sub[["x_sim","y_sim","z_sim"]].values, axis=1).mean()
            ml_err = np.linalg.norm(sub[["x","y","z"]].values - sub[["x_pred","y_pred","z_pred"]].values, axis=1).mean()
            
            print(f"--- Sat {sid} Error Analysis ---")
            print(f"Base Simulation Error: {base_err:.3f} km")
            print(f"ML Corrected Error:    {ml_err:.3f} km")
            print(f"Improvement:           {(base_err - ml_err):.3f} km")
        
    print("\nDone.")

if __name__ == "__main__":
    main()