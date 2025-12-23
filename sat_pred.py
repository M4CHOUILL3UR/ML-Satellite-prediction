import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Helper to print df shape/head quickly
def show_df(df, name="DataFrame", n=10):
    print(f"\n--- {name} (shape={df.shape}) ---")
    print(df.head(n).to_string(index=False))

def load_data(base_path):
    # Load raw csvs
    print(f"Loading data from {base_path}...")
    train_df = pd.read_csv(os.path.join(base_path, "jan_train.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "jan_test.csv"))
    answer_df = pd.read_csv(os.path.join(base_path, "answer_key.csv"))

    # Select relevant cols for training
    train_table = train_df[[
        "id", "sat_id",
        "x", "y", "z", "Vx", "Vy", "Vz",
        "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"
    ]].copy()
    train_table["set"] = "train"

    # Merge answer key into test for evaluation later
    pred_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    test_preds = test_df.copy().reset_index(drop=True)
    test_preds[pred_cols] = answer_df[pred_cols].values
    test_preds["set"] = "test_pred"

    # specific use case: merging all for visualization context
    merged_all = pd.concat([train_table, test_preds], ignore_index=True)

    return train_df, test_df, answer_df, train_table, test_preds, merged_all

# Basic geometric cut to isolate one orbit
def cut_after_one_orbit_3D(df):
    df = df.copy()
    cols = ["x", "y", "z", "Vx", "Vy", "Vz"]

    # cleanup duplicates if any exist
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # ensure numeric
    for c in cols:
        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df[c] = pd.to_numeric(col, errors="coerce")

    df = df.dropna(subset=cols)
    if df.shape[0] < 5:
        return df

    # Vector math to find orbit plane
    r = df[["x", "y", "z"]].to_numpy(dtype=float)
    v = df[["Vx", "Vy", "Vz"]].to_numpy(dtype=float)

    hv = np.cross(r, v)
    h = np.nanmean(hv, axis=0)
    hn = np.linalg.norm(h)
    
    # Filter weird data
    if not np.isfinite(hn) or hn < 1e-12:
        return df

    h = h / hn
    
    # arbitrary axis setup
    u_axis = np.cross(h, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(u_axis) < 1e-6:
        u_axis = np.cross(h, np.array([0.0, 1.0, 0.0]))
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(h, u_axis)

    u = r @ u_axis
    w = r @ v_axis
    theta = np.unwrap(np.arctan2(w, u))

    # Cut when theta completes a circle
    theta0 = theta[0]
    mask = (theta - theta0) <= 2 * np.pi
    return df.loc[mask].copy()

def compute_train_metrics(train_table, out_csv_path):
    metrics = []
    # iterating per sat to get individual errors
    for sat in train_table["sat_id"].unique():
        subset = train_table[train_table["sat_id"] == sat]

        pos_rmse = np.sqrt(np.mean(
            (subset[["x", "y", "z"]].values - subset[["x_sim", "y_sim", "z_sim"]].values) ** 2
        ))
        vel_rmse = np.sqrt(np.mean(
            (subset[["Vx", "Vy", "Vz"]].values - subset[["Vx_sim", "Vy_sim", "Vz_sim"]].values) ** 2
        ))

        metrics.append({"sat_id": sat, "pos_RMSE": pos_rmse, "vel_RMSE": vel_rmse})

    metrics_df = pd.DataFrame(metrics).sort_values("sat_id")
    metrics_df.to_csv(out_csv_path, index=False)
    return metrics_df

def plot_matplotlib_3d_sample(merged_all, nb_sats=15, R_earth=6371):
    # filter for usable satellites
    valid_sats = [
        sat for sat in merged_all["sat_id"].unique()
        if merged_all.loc[merged_all["sat_id"] == sat, ["x", "y", "z"]].dropna().shape[0] > 20
    ]
    if not valid_sats:
        print("No valid satellites for matplotlib plot.")
        return

    sample_sats = np.random.choice(valid_sats, size=min(nb_sats, len(valid_sats)), replace=False)
    print(f"\nPlotting sample (matplotlib): {sample_sats}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe earth
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="lightblue", alpha=0.35, linewidth=0)

    colors = plt.cm.tab20(np.linspace(0, 1, len(sample_sats)))

    for i, sat in enumerate(sample_sats):
        sub = merged_all[merged_all["sat_id"] == sat].copy()
        one_turn = cut_after_one_orbit_3D(sub)
        ax.plot(one_turn["x"], one_turn["y"], one_turn["z"],
                color=colors[i], linewidth=2, label=f"sat {sat}")

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_plotly_real_vs_sim(train_table, nb_sats=10, R_earth=6371, open_in_browser=False):
    valid_sats = train_table["sat_id"].unique()
    if len(valid_sats) == 0:
        return

    sample_sats = np.random.choice(valid_sats, size=min(nb_sats, len(valid_sats)), replace=False)
    
    fig = go.Figure()
    
    # sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))
    fig.add_surface(x=xs, y=ys, z=zs, colorscale="Blues", opacity=0.9, showscale=False)

    real_colors = px.colors.qualitative.Dark24

    for i, sat in enumerate(sample_sats):
        sub = train_table[train_table["sat_id"] == sat].copy()
        real = cut_after_one_orbit_3D(sub)
        
        c_real = real_colors[i % len(real_colors)]
        
        # Real path
        fig.add_trace(go.Scatter3d(
            x=real["x"], y=real["y"], z=real["z"],
            mode="lines", line=dict(color=c_real, width=6),
            name=f"Sat {sat} (Real)"
        ))

        # Sim path check
        sim_df = sub[["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]].copy()
        sim_df.columns = ["x", "y", "z", "Vx", "Vy", "Vz"]
        
        # basic cleaning
        for c in ["x", "y", "z", "Vx", "Vy", "Vz"]:
            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce")
        sim_df = sim_df.dropna()

        sim_orbit = cut_after_one_orbit_3D(sim_df)

        fig.add_trace(go.Scatter3d(
            x=sim_orbit["x"], y=sim_orbit["y"], z=sim_orbit["z"],
            mode="lines",
            line=dict(color='white', width=3, dash="dash"), # simplified color logic
            opacity=0.7,
            name=f"Sat {sat} (Sim)"
        ))

    fig.update_layout(template="plotly_dark", scene=dict(aspectmode="data"))
    if open_in_browser:
        import plotly.io as pio
        pio.renderers.default = "browser"
    fig.show()

def add_features(df):
    df = df.copy()
    # Basic kinematics
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["speed"] = np.sqrt(df["Vx"]**2 + df["Vy"]**2 + df["Vz"]**2)
    df["altitude"] = df["r"] - 6371.0
    df["theta_xy"] = np.arctan2(df["y"], df["x"])

    # Angular momentum
    hx = df["y"]*df["Vz"] - df["z"]*df["Vy"]
    hy = df["z"]*df["Vx"] - df["x"]*df["Vz"]
    hz = df["x"]*df["Vy"] - df["y"]*df["Vx"]
    df["h_norm"] = np.sqrt(hx**2 + hy**2 + hz**2)
    df["incl"] = np.degrees(np.arccos(hz / (df["h_norm"] + 1e-9)))
    return df

def train_rf_direct_model(df_ml, feature_cols, target_cols, n_samples=5000):
    # Predict sim from real directly
    df_ml = df_ml.dropna(subset=target_cols).copy()
    
    # Subsampling for speed
    n_samples_eff = min(n_samples, df_ml.shape[0])
    df_s = df_ml.sample(n_samples_eff, random_state=42).copy()

    X = df_s[feature_cols]
    y = df_s[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base RF model
    rf = RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n[Direct Model] RMSE: {rmse:.3f}")

    return rf, scaler, (X_test, y_test, y_pred), df_s

def train_rf_correction_model(df_s, feature_cols):
    # Predict the residual (real - sim)
    df_corr = df_s.copy()
    for comp in ["x", "y", "z", "Vx", "Vy", "Vz"]:
        df_corr[f"d_{comp}"] = df_corr[comp] - df_corr[f"{comp}_sim"]

    target_corr_cols = ["d_x", "d_y", "d_z", "d_Vx", "d_Vy", "d_Vz"]

    X_corr = df_corr[feature_cols]
    y_corr = df_corr[target_corr_cols]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_corr, y_corr, test_size=0.2, random_state=42)

    scaler_corr = StandardScaler()
    Xc_train_scaled = scaler_corr.fit_transform(Xc_train)
    Xc_test_scaled = scaler_corr.transform(Xc_test)

    rf_corr = RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=42)
    rf_corr.fit(Xc_train_scaled, yc_train)
    
    print("\n[Correction Model] Trained.")
    return rf_corr, scaler_corr

def compare_one_satellite_plotly(train_fe, sat, feature_cols, rf_direct, scaler_direct, rf_corr, scaler_corr):
    sub = train_fe[train_fe["sat_id"] == sat].copy()

    # Real orbit
    real_orbit = cut_after_one_orbit_3D(sub)

    # Sim orbit
    sim_df = sub.dropna(subset=["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]).copy()
    sim_df.rename(columns={
        "x_sim": "x", "y_sim": "y", "z_sim": "z",
        "Vx_sim": "Vx", "Vy_sim": "Vy", "Vz_sim": "Vz",
    }, inplace=True)
    sim_orbit = cut_after_one_orbit_3D(sim_df)

    # ML direct
    X_sat = scaler_direct.transform(sub[feature_cols])
    y_pred_sat = rf_direct.predict(X_sat)
    ml_df = pd.DataFrame(y_pred_sat, columns=["x", "y", "z", "Vx", "Vy", "Vz"])
    ml_orbit = cut_after_one_orbit_3D(ml_df)

    # ML correction
    Xc_sat = scaler_corr.transform(sub[feature_cols])
    d_pred = rf_corr.predict(Xc_sat)
    d_df = pd.DataFrame(d_pred, columns=["d_x", "d_y", "d_z", "d_Vx", "d_Vy", "d_Vz"])

    # Corrected = Sim + Predicted_Residual
    corr_df = pd.DataFrame({
        "x": sub["x_sim"].values + d_df["d_x"].values,
        "y": sub["y_sim"].values + d_df["d_y"].values,
        "z": sub["z_sim"].values + d_df["d_z"].values,
    }).dropna()
    # Need velocities for the cutter to work, hacking them in
    corr_df["Vx"] = 0 
    corr_df["Vy"] = 0
    corr_df["Vz"] = 0
    
    # Just plot raw points for correction to avoid the complex cutter failing on bad velocity data
    # ...

    fig = go.Figure()
    
    # Real
    fig.add_trace(go.Scatter3d(x=real_orbit["x"], y=real_orbit["y"], z=real_orbit["z"],
                               mode="lines", name="Real", line=dict(color="cyan", width=5)))
    # Sim
    fig.add_trace(go.Scatter3d(x=sim_orbit["x"], y=sim_orbit["y"], z=sim_orbit["z"],
                               mode="lines", name="Sim", line=dict(color="yellow", width=3, dash="dash")))
    # Direct ML
    fig.add_trace(go.Scatter3d(x=ml_orbit["x"], y=ml_orbit["y"], z=ml_orbit["z"],
                               mode="lines", name="ML Direct", line=dict(color="magenta", width=3)))
    
    # Correction
    # plotting all points instead of cut orbit for safety
    fig.add_trace(go.Scatter3d(x=corr_df["x"], y=corr_df["y"], z=corr_df["z"],
                               mode="markers", name="ML Corrected", marker=dict(color="lime", size=2)))

    fig.update_layout(title=f"Sat {sat} Comparison", template="plotly_dark")
    fig.show()

def main():
    # Use current directory
    path = os.getcwd()

    train_df, test_df, answer_df, train_table, test_preds, merged_all = load_data(path)
    show_df(merged_all, "Merged Data")

    # Save metrics
    metrics_path = os.path.join(path, "satellite_metrics.csv")
    compute_train_metrics(train_table, metrics_path)

    # Visualization
    plot_matplotlib_3d_sample(merged_all)
    plot_plotly_real_vs_sim(train_table)

    # ML Prep
    train_fe = add_features(train_table)
    
    feats = ["x", "y", "z", "Vx", "Vy", "Vz", "r", "speed", "altitude", "theta_xy", "h_norm", "incl"]
    targets = ["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]

    df_ml = train_fe.dropna(subset=targets).copy()

    # Train
    rf_direct, scaler_direct, _, df_sampled = train_rf_direct_model(df_ml, feats, targets)
    rf_corr, scaler_corr = train_rf_correction_model(df_sampled, feats)

    # Compare
    test_sat = int(df_sampled["sat_id"].iloc[0])
    compare_one_satellite_plotly(train_fe, test_sat, feats, rf_direct, scaler_direct, rf_corr, scaler_corr)

if __name__ == "__main__":
    main()