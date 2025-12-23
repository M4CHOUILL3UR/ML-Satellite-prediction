# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 18:17:37 2025

@author: mathi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D Matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 0) UTILITIES (Spyder-safe display)
# ============================================================

def show_df(df, name="DataFrame", n=10):
    """Spyder-friendly DataFrame preview."""
    print(f"\n--- {name} (shape={df.shape}) ---")
    print(df.head(n).to_string(index=False))


# ============================================================
# 1) DATA LOADING (your requested "data harvest" style)
# ============================================================

def load_data(path):
    """
    Load jan_train / jan_test, then inject answer_key columns into test as predictions.
    Returns:
      train_df (raw)
      test_df  (raw)
      answer_df
      train_table (formatted train subset + set='train')
      test_preds  (test with injected predictions + set='test_pred')
      merged_all  (concat train_table + test_preds)
    """
    train_df = pd.read_csv(os.path.join(path, "jan_train.csv"))
    test_df = pd.read_csv(os.path.join(path, "jan_test.csv"))
    answer_df = pd.read_csv(os.path.join(path, "answer_key.csv"))

    # If id exists, keep it for traceability (you used it later)
    # Train table with required columns
    train_table = train_df[[
        "id", "sat_id",
        "x", "y", "z", "Vx", "Vy", "Vz",
        "x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"
    ]].copy()
    train_table["set"] = "train"

    # Inject answer into test to create "test_pred"
    pred_cols = ["x", "y", "z", "Vx", "Vy", "Vz"]
    test_preds = test_df.copy().reset_index(drop=True)
    test_preds[pred_cols] = answer_df[pred_cols].values
    test_preds["set"] = "test_pred"

    merged_all = pd.concat([train_table, test_preds], ignore_index=True)

    return train_df, test_df, answer_df, train_table, test_preds, merged_all


# ============================================================
# 2) ORBIT CUTTER
# ============================================================

def cut_after_one_orbit_3D(df):
    df = df.copy()

    cols = ["x", "y", "z", "Vx", "Vy", "Vz"]

    # --- Fix: handle duplicated column names (x, y, etc.) ---
    if df.columns.duplicated().any():
        # keep the first occurrence of each column name
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Force numeric robustly (even if df[c] accidentally returns a DataFrame)
    for c in cols:
        col = df[c]
        if isinstance(col, pd.DataFrame):   # duplicated columns case
            col = col.iloc[:, 0]
        df[c] = pd.to_numeric(col, errors="coerce")

    df = df.dropna(subset=cols)
    if df.shape[0] < 5:
        return df

    r = df[["x", "y", "z"]].to_numpy(dtype=float)
    v = df[["Vx", "Vy", "Vz"]].to_numpy(dtype=float)

    hv = np.cross(r, v)
    h = np.nanmean(hv, axis=0)
    hn = np.linalg.norm(h)
    if not np.isfinite(hn) or hn < 1e-12:
        return df

    h = h / hn

    u_axis = np.cross(h, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(u_axis) < 1e-6:
        u_axis = np.cross(h, np.array([0.0, 1.0, 0.0]))
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(h, u_axis)

    u = r @ u_axis
    w = r @ v_axis
    theta = np.unwrap(np.arctan2(w, u))

    theta0 = theta[0]
    mask = (theta - theta0) <= 2 * np.pi
    return df.loc[mask].copy()



# ============================================================
# 3) METRICS (train: real vs sim)
# ============================================================

def compute_train_metrics(train_table, out_csv_path):
    metrics = []
    for sat in train_table["sat_id"].unique():
        subset = train_table[train_table["sat_id"] == sat]

        # position RMSE (aggregate over all coords & timesteps)
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


# ============================================================
# 4) PLOTS
# ============================================================

def plot_matplotlib_3d_sample(merged_all, nb_sats=15, R_earth=6371):
    valid_sats = [
        sat for sat in merged_all["sat_id"].unique()
        if merged_all.loc[merged_all["sat_id"] == sat, ["x", "y", "z"]].dropna().shape[0] > 20
    ]
    if len(valid_sats) == 0:
        print("No valid satellites to plot (matplotlib).")
        return

    sample_sats = np.random.choice(valid_sats, size=min(nb_sats, len(valid_sats)), replace=False)
    print(f"\nSatellites plotted (matplotlib, 1 orbit each): {sample_sats}")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

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

    ax.set_title("3D trajectories — 1 orbital turn per satellite")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()


def plot_plotly_real_vs_sim(train_table, nb_sats=10, R_earth=6371, open_in_browser=False):
    valid_sats = train_table["sat_id"].unique()
    if len(valid_sats) == 0:
        print("No satellites to plot (plotly).")
        return

    sample_sats = np.random.choice(valid_sats, size=min(nb_sats, len(valid_sats)), replace=False)
    print(f"\nSatellites selected (plotly real vs sim): {sample_sats}")

    fig = go.Figure()

    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R_earth * np.outer(np.cos(u), np.sin(v))
    ys = R_earth * np.outer(np.sin(u), np.sin(v))
    zs = R_earth * np.outer(np.ones_like(u), np.cos(v))

    fig.add_surface(x=xs, y=ys, z=zs, colorscale="Blues", opacity=0.9, showscale=False)

    real_colors = px.colors.qualitative.Dark24

    def invert_color(hex_color):
        hex_color = hex_color.lstrip('#')
        r = 255 - int(hex_color[0:2], 16)
        g = 255 - int(hex_color[2:4], 16)
        b = 255 - int(hex_color[4:6], 16)
        return f"rgb({r},{g},{b})"

    for i, sat in enumerate(sample_sats):
        sub = train_table[train_table["sat_id"] == sat].copy()

        real = cut_after_one_orbit_3D(sub)
        color_real = real_colors[i % len(real_colors)]
        color_sim = invert_color(color_real.replace(" ", ""))

        fig.add_trace(go.Scatter3d(
            x=real["x"], y=real["y"], z=real["z"],
            mode="lines",
            line=dict(color=color_real, width=6),
            name=f"Satellite {sat} — Real"
        ))

        if sub[["x_sim", "y_sim", "z_sim"]].dropna().shape[0] > 20:

            # Sim orbit (rename to generic columns for cutter)
            sim_df = sub[["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]].copy()
            sim_df = sim_df.rename(columns={
                "x_sim": "x", "y_sim": "y", "z_sim": "z",
                "Vx_sim": "Vx", "Vy_sim": "Vy", "Vz_sim": "Vz",
            })

        # Force numeric + drop NaNs (crucial)
        for c in ["x", "y", "z", "Vx", "Vy", "Vz"]:
            sim_df[c] = pd.to_numeric(sim_df[c], errors="coerce")
        sim_df = sim_df.dropna(subset=["x", "y", "z", "Vx", "Vy", "Vz"])

        sim_orbit = cut_after_one_orbit_3D(sim_df)

        fig.add_trace(go.Scatter3d(
            x=sim_orbit["x"], y=sim_orbit["y"], z=sim_orbit["z"],
            mode="lines",
            line=dict(color=color_sim, width=4, dash="dash"),
            opacity=0.9,
            name=f"Satellite {sat} — Simulation"
        ))

    fig.update_layout(
        title="Interactive 3D — Real & Simulated Orbits",
        template="plotly_dark",
        legend=dict(font=dict(size=12)),
        scene=dict(
            xaxis_title="x (km)",
            yaxis_title="y (km)",
            zaxis_title="z (km)",
            aspectmode="data",
        )
    )

    if open_in_browser:
        import plotly.io as pio
        pio.renderers.default = "browser"

    fig.show()


# ============================================================
# 5) FEATURE ENGINEERING
# ============================================================

def add_features(df):
    df = df.copy()
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["speed"] = np.sqrt(df["Vx"]**2 + df["Vy"]**2 + df["Vz"]**2)
    df["altitude"] = df["r"] - 6371.0
    df["theta_xy"] = np.arctan2(df["y"], df["x"])

    hx = df["y"]*df["Vz"] - df["z"]*df["Vy"]
    hy = df["z"]*df["Vx"] - df["x"]*df["Vz"]
    hz = df["x"]*df["Vy"] - df["y"]*df["Vx"]
    df["h_norm"] = np.sqrt(hx**2 + hy**2 + hz**2)
    df["incl"] = np.degrees(np.arccos(hz / (df["h_norm"] + 1e-9)))
    return df


# ============================================================
# 6) ML MODELS
# ============================================================

def train_rf_direct_model(df_ml, feature_cols, target_cols, n_samples=5000, random_state=42):
    """
    Train a light RandomForest that predicts SIM state from REAL state + engineered features.
    Returns: rf_model, scaler, (X_test, y_test, y_pred)
    """
    df_ml = df_ml.dropna(subset=target_cols).copy()
    if df_ml.shape[0] == 0:
        raise ValueError("No rows available for ML after dropping NaNs in targets.")

    n_samples_eff = min(n_samples, df_ml.shape[0])
    df_s = df_ml.sample(n_samples_eff, random_state=random_state).copy()

    X = df_s[feature_cols]
    y = df_s[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=60,
        max_depth=12,
        n_jobs=-1,
        random_state=random_state
    )

    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Random Forest DIRECT model (predict sim) ===")
    print(f"Samples used: {n_samples_eff}")
    print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.4f}")

    return rf, scaler, (X_test, y_test, y_pred), df_s


def train_rf_correction_model(df_s, feature_cols, random_state=42):
    """
    Train RF to predict residuals: (real - sim).
    Requires df_s to include columns:
      real: x,y,z,Vx,Vy,Vz
      sim:  x_sim,y_sim,z_sim,Vx_sim,Vy_sim,Vz_sim
    Returns: rf_corr, scaler_corr, (Xc_test, yc_test, yc_pred)
    """
    df_corr = df_s.copy()
    for comp in ["x", "y", "z", "Vx", "Vy", "Vz"]:
        df_corr[f"d_{comp}"] = df_corr[comp] - df_corr[f"{comp}_sim"]

    target_corr_cols = ["d_x", "d_y", "d_z", "d_Vx", "d_Vy", "d_Vz"]

    X_corr = df_corr[feature_cols]
    y_corr = df_corr[target_corr_cols]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_corr, y_corr, test_size=0.2, random_state=random_state
    )

    scaler_corr = StandardScaler()
    Xc_train_scaled = scaler_corr.fit_transform(Xc_train)
    Xc_test_scaled = scaler_corr.transform(Xc_test)

    rf_corr = RandomForestRegressor(
        n_estimators=60,
        max_depth=12,
        n_jobs=-1,
        random_state=random_state
    )

    rf_corr.fit(Xc_train_scaled, yc_train)
    yc_pred = rf_corr.predict(Xc_test_scaled)

    rmse_corr = np.sqrt(mean_squared_error(yc_test, yc_pred))
    mae_corr = mean_absolute_error(yc_test, yc_pred)

    print("\n=== Random Forest CORRECTION model (predict residual real - sim) ===")
    print(f"RMSE residuals: {rmse_corr:.3f} | MAE residuals: {mae_corr:.3f}")

    return rf_corr, scaler_corr, (Xc_test, yc_test, yc_pred)


# ============================================================
# 7) SINGLE SATELLITE COMPARISON (Real vs Sim vs ML direct vs ML corr)
# ============================================================

def compare_one_satellite_plotly(train_fe, sat, feature_cols, rf_direct, scaler_direct, rf_corr, scaler_corr, open_in_browser=False):
    sub = train_fe[train_fe["sat_id"] == sat].copy()

    # Real orbit
    real_orbit = cut_after_one_orbit_3D(sub)

    # Sim orbit (rename to generic columns for cutter)
    sim_df = sub.dropna(subset=["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]).copy()
    sim_df = sim_df.rename(columns={
        "x_sim": "x", "y_sim": "y", "z_sim": "z",
        "Vx_sim": "Vx", "Vy_sim": "Vy", "Vz_sim": "Vz",
    })
    dups = sim_df.columns[sim_df.columns.duplicated()].tolist()
    print("Duplicate columns in sim_df:", dups)
    print("sim_df columns:", list(sim_df.columns))

    sim_orbit = cut_after_one_orbit_3D(sim_df)

    # ML direct sim prediction
    X_sat = scaler_direct.transform(sub[feature_cols])
    y_pred_sat = rf_direct.predict(X_sat)
    ml_df = pd.DataFrame(y_pred_sat, columns=["x", "y", "z", "Vx", "Vy", "Vz"])
    ml_orbit = cut_after_one_orbit_3D(ml_df)

    # ML correction (sim + residual)
    Xc_sat = scaler_corr.transform(sub[feature_cols])
    d_pred = rf_corr.predict(Xc_sat)
    d_df = pd.DataFrame(d_pred, columns=["d_x", "d_y", "d_z", "d_Vx", "d_Vy", "d_Vz"])

    # Build corrected trajectory: sim + d
    corr_df = pd.DataFrame({
        "x": sub["x_sim"].values + d_df["d_x"].values,
        "y": sub["y_sim"].values + d_df["d_y"].values,
        "z": sub["z_sim"].values + d_df["d_z"].values,
        "Vx": sub["Vx_sim"].values + d_df["d_Vx"].values,
        "Vy": sub["Vy_sim"].values + d_df["d_Vy"].values,
        "Vz": sub["Vz_sim"].values + d_df["d_Vz"].values,
    }).dropna()
    corr_orbit = cut_after_one_orbit_3D(corr_df)

    # Earth
    R = 6371
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(x=xs, y=ys, z=zs, colorscale="Blues", opacity=0.85, showscale=False)

    fig.add_trace(go.Scatter3d(
        x=real_orbit["x"], y=real_orbit["y"], z=real_orbit["z"],
        mode="lines", name="Real Orbit",
        line=dict(color="cyan", width=6)
    ))

    fig.add_trace(go.Scatter3d(
        x=sim_orbit["x"], y=sim_orbit["y"], z=sim_orbit["z"],
        mode="lines", name="Simulated Orbit",
        line=dict(color="yellow", width=4, dash="dash")
    ))

    fig.add_trace(go.Scatter3d(
        x=ml_orbit["x"], y=ml_orbit["y"], z=ml_orbit["z"],
        mode="lines", name="ML Direct (sim)",
        line=dict(color="magenta", width=4)
    ))

    fig.add_trace(go.Scatter3d(
        x=corr_orbit["x"], y=corr_orbit["y"], z=corr_orbit["z"],
        mode="lines", name="ML Correction (sim + residual)",
        line=dict(color="lime", width=4)
    ))

    fig.update_layout(
        title=f"3D Comparison — Satellite {sat}",
        template="plotly_dark",
        scene=dict(
            xaxis_title="x (km)",
            yaxis_title="y (km)",
            zaxis_title="z (km)",
            aspectmode="data"
        ),
        legend=dict(font=dict(size=12))
    )

    if open_in_browser:
        import plotly.io as pio
        pio.renderers.default = "browser"

    fig.show()

    # Also compute error curves (SIM vs DIRECT vs CORR)
    # Build aligned table using sub index
    out = pd.DataFrame({
        "x_real": sub["x"].values, "y_real": sub["y"].values, "z_real": sub["z"].values,
        "x_sim": sub["x_sim"].values, "y_sim": sub["y_sim"].values, "z_sim": sub["z_sim"].values,
        "x_dir": ml_df["x"].values, "y_dir": ml_df["y"].values, "z_dir": ml_df["z"].values,
        "x_corr": (sub["x_sim"].values + d_df["d_x"].values),
        "y_corr": (sub["y_sim"].values + d_df["d_y"].values),
        "z_corr": (sub["z_sim"].values + d_df["d_z"].values),
    }).dropna()

    out["err_sim_3D"] = np.sqrt((out["x_sim"]-out["x_real"])**2 + (out["y_sim"]-out["y_real"])**2 + (out["z_sim"]-out["z_real"])**2)
    out["err_dir_3D"] = np.sqrt((out["x_dir"]-out["x_real"])**2 + (out["y_dir"]-out["y_real"])**2 + (out["z_dir"]-out["z_real"])**2)
    out["err_corr_3D"] = np.sqrt((out["x_corr"]-out["x_real"])**2 + (out["y_corr"]-out["y_real"])**2 + (out["z_corr"]-out["z_real"])**2)

    print("\n=== Error summary (this satellite) ===")
    print(pd.DataFrame({
        "Model": ["Simulation", "ML direct", "ML correction"],
        "Mean 3D pos error (km)": [out["err_sim_3D"].mean(), out["err_dir_3D"].mean(), out["err_corr_3D"].mean()],
        "Max 3D pos error (km)":  [out["err_sim_3D"].max(),  out["err_dir_3D"].max(),  out["err_corr_3D"].max()],
    }).to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.plot(out["err_sim_3D"].values, label="Simulation — 3D error", alpha=0.85)
    plt.plot(out["err_dir_3D"].values, label="ML direct — 3D error", alpha=0.75)
    plt.plot(out["err_corr_3D"].values, label="ML correction — 3D error", alpha=0.75)
    plt.xlabel("Time step")
    plt.ylabel("Position error (km)")
    plt.title(f"3D position error over time — Satellite {sat}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    # ---- USER PATH ----
    path = r"C:\Users\mathi\OneDrive\Bureau\archive"

    # ---- LOAD ----
    train_df, test_df, answer_df, train_table, test_preds, merged_all = load_data(path)

    print("=== Merged table: train + test + injected predictions ===")
    show_df(merged_all, "merged_all", n=20)

    # ---- METRICS (real vs sim on train) ----
    metrics_path = os.path.join(path, "satellite_metrics.csv")
    metrics_df = compute_train_metrics(train_table, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")
    show_df(metrics_df, "metrics_df", n=10)

    # ---- PLOTS ----
    plot_matplotlib_3d_sample(merged_all, nb_sats=15, R_earth=6371)

    # Plotly can be set to open in browser if Spyder renderer is tricky
    plot_plotly_real_vs_sim(train_table, nb_sats=10, R_earth=6371, open_in_browser=False)

    # ---- FEATURE ENGINEERING ----
    train_fe = add_features(train_table)

    feature_cols = [
        "x", "y", "z", "Vx", "Vy", "Vz",
        "r", "speed", "altitude", "theta_xy", "h_norm", "incl"
    ]
    target_cols = ["x_sim", "y_sim", "z_sim", "Vx_sim", "Vy_sim", "Vz_sim"]

    df_ml = train_fe.dropna(subset=target_cols).copy()

    # ---- ML DIRECT MODEL ----
    rf_direct, scaler_direct, (X_test, y_test, y_pred), df_sampled = train_rf_direct_model(
        df_ml=df_ml,
        feature_cols=feature_cols,
        target_cols=target_cols,
        n_samples=5000,
        random_state=42
    )

    # ---- ML CORRECTION MODEL ----
    rf_corr, scaler_corr, _ = train_rf_correction_model(
        df_s=df_sampled,
        feature_cols=feature_cols,
        random_state=42
    )

    # ---- COMPARE ONE SATELLITE ----
    sat = int(df_sampled["sat_id"].sample(1, random_state=42).iloc[0])
    print(f"\nSelected satellite for detailed comparison: {sat}")

    compare_one_satellite_plotly(
        train_fe=train_fe,
        sat=sat,
        feature_cols=feature_cols,
        rf_direct=rf_direct,
        scaler_direct=scaler_direct,
        rf_corr=rf_corr,
        scaler_corr=scaler_corr,
        open_in_browser=False
    )


if __name__ == "__main__":
    main()
