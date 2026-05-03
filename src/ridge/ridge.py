"""
=============================================================
  ENSO (Niño 3.4) Forecasting with Ridge Regression
=============================================================
  Features  : Niño 3.4, SOI, OLR anomaly, Indian Ocean SST
  Target    : Niño 3.4 index at lead times 1, 3, 6, 9, 12, 15 months
  Model     : Ridge Regression (scikit-learn)
  Plots     : Interactive HTML via Plotly
=============================================================
"""

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Color palette: one distinct color per lead time ──────────────────────────
LEAD_COLORS = {
    1:  "#FF5722",
    3:  "#E91E63",
    6:  "#9C27B0",
    9:  "#2196F3",
    12: "#00BCD4",
    15: "#4CAF50",
}

DARK_BG    = "#0D1117"
PANEL_BG   = "#161B22"
GRID_COLOR = "#30363D"
TEXT_COLOR = "#C9D1D9"
ACTUAL_CLR = "#58A6FF"

def _plotly_dark_layout(**extra):
    """Base Plotly layout for consistent dark theme across all figures."""
    base = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="monospace"),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=GRID_COLOR, borderwidth=1),
        hoverlabel=dict(bgcolor="#21262D", font_size=12),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    base.update(extra)
    return base


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – DATA LOADING 
# ══════════════════════════════════════════════════════════════════════════════

def load_nino34(path):
    df = pd.read_csv(path, sep=r"\s+", header=0, engine="python")
    last_col = df.columns[-1]
    df = df[["YR", "MON", last_col]].copy()
    df.rename(columns={"YR": "year", "MON": "month", last_col: "nino34"}, inplace=True)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    return df[["time", "nino34"]].sort_values("time").reset_index(drop=True)


_TOKEN_RE = re.compile(r"-?\d+\.?\d*")

def _parse_noaa_block(path, col_name, block_index):
    with open(path) as f:
        lines = f.readlines()

    found = -1
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("YEAR"):
            found += 1
            if found == block_index:
                data_start = i + 1
                break

    if data_start is None:
        raise ValueError(f"Block {block_index} not found in {path}")

    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            break
        tokens = _TOKEN_RE.findall(line)
        if not tokens or not tokens[0].isdigit():
            break
        if len(tokens) < 13:
            continue
        year = int(tokens[0])
        for month, tok in enumerate(tokens[1:13], start=1):
            val = float(tok)
            if val <= -999.0:
                val = np.nan
            rows.append({"year": year, "month": month, col_name: val})

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df[["time", col_name]].sort_values("time").reset_index(drop=True)


def load_soi(path):
    return _parse_noaa_block(path, "soi", block_index=0)

def load_olr(path):
    return _parse_noaa_block(path, "olr", block_index=1)

def load_sst_india(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df.rename(columns={"thetao": "sst_india"}, inplace=True)
    df["time"] = df["time"].dt.to_period("M").dt.to_timestamp()
    return df[["time", "sst_india"]].sort_values("time").reset_index(drop=True)


def load_and_merge(nino_path, soi_path, olr_path, sst_path):
    print("Loading datasets")
    nino = load_nino34(nino_path)
    soi  = load_soi(soi_path)
    olr  = load_olr(olr_path)
    sst  = load_sst_india(sst_path)

    df = (nino
          .merge(soi, on="time", how="outer")
          .merge(olr, on="time", how="outer")
          .merge(sst, on="time", how="outer"))

    df = df.sort_values("time").reset_index(drop=True)
    df.replace(-999.9, np.nan, inplace=True)

    print(f"   Merged shape (before cleaning): {df.shape}")
    print(f"   Time range : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"   Missing per column:\n{df.isnull().sum()}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – FEATURE ENGINEERING  
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["nino34", "soi", "olr", "sst_india"]
LAG_MONTHS   = [1, 2, 3, 4, 5, 6]


def build_dataset(df, lead, lags=LAG_MONTHS):
    df = df.copy()
    lag_names = []
    for col in FEATURE_COLS:
        for lag in lags:
            name = f"{col}_lag{lag}"
            df[name] = df[col].shift(lag)
            lag_names.append(name)

    df["target"] = df["nino34"].shift(-lead)
    df = df[["time"] + lag_names + ["target"]].dropna().reset_index(drop=True)

    X     = df[lag_names].values
    y     = df["target"].values
    times = df["time"]
    return X, y, times, lag_names


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – TRAIN / TEST SPLIT  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def time_split(X, y, times, train_frac=0.80):
    n      = len(y)
    cutoff = int(n * train_frac)
    return (X[:cutoff], X[cutoff:],
            y[:cutoff], y[cutoff:],
            times.iloc[:cutoff], times.iloc[cutoff:])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – MODEL  ]
# ══════════════════════════════════════════════════════════════════════════════

def train_ridge(X_tr, y_tr, alpha=1.0):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    model  = Ridge(alpha=alpha)
    model.fit(X_tr_s, y_tr)
    return scaler, model


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – EVALUATION 
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, lead):
    rmse       = np.sqrt(mean_squared_error(y_true, y_pred))
    mae        = mean_absolute_error(y_true, y_pred)
    corr, pval = pearsonr(y_true, y_pred)
    return {"lead": lead, "rmse": rmse, "mae": mae, "corr": corr, "pval": pval}


def print_metrics_table(results):
    print("\n" + "═" * 62)
    print(f"  {'Lead':>5}  {'RMSE':>8}  {'MAE':>8}  {'Corr':>8}  {'p-value':>10}")
    print("─" * 62)
    for r in results:
        print(f"  {r['lead']:>4}m  {r['rmse']:>8.4f}  {r['mae']:>8.4f}"
              f"  {r['corr']:>8.4f}  {r['pval']:>10.2e}")
    print("═" * 62 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 – INTERACTIVE VISUALISATION 
# ══════════════════════════════════════════════════════════════════════════════

def _lead_color(lead):
    """Return a color for the given lead time, cycling if unlisted."""
    fallback = ["#FF5722", "#E91E63", "#9C27B0", "#2196F3", "#00BCD4", "#4CAF50",
                "#FF9800", "#00E5FF", "#76FF03", "#EA80FC"]
    return LEAD_COLORS.get(lead, fallback[lead % len(fallback)])


def plot_predictions(results_dict, out_dir):
    """
    One subplot per lead time, each showing:
      - Actual Niño 3.4 (shaded + line)
      - Predicted Niño 3.4 (dashed line)
    Hover shows: time | actual | predicted
    Saved to enso_predictions.html
    """
    leads = sorted(results_dict.keys())
    n     = len(leads)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        subplot_titles=[
            f"ENSO Forecast — Lead = {l} month{'s' if l > 1 else ''}  "
            f"[RMSE={results_dict[l]['metrics']['rmse']:.3f}  "
            f"Corr={results_dict[l]['metrics']['corr']:.3f}]"
            for l in leads
        ],
        vertical_spacing=0.06,
    )

    for row, lead in enumerate(leads, start=1):
        r      = results_dict[lead]
        t      = r["times_test"]
        y_true = r["y_true"]
        y_pred = r["y_pred"]
        color  = _lead_color(lead)

        hover_actual = [
            f"<b>Time:</b> {ts.strftime('%Y-%m')}<br>"
            f"<b>Actual:</b> {a:.3f} °C<br>"
            f"<b>Predicted:</b> {p:.3f} °C"
            for ts, a, p in zip(t, y_true, y_pred)
        ]
        hover_pred = hover_actual 

        # Shaded El Niño / La Niña bands
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(np.where(y_true >= 0.5, y_true, 0.5)) + [0.5] * len(t),
            fill="toself", fillcolor="rgba(255,87,34,0.12)",
            line=dict(width=0), showlegend=(row == 1),
            name="El Niño (≥ +0.5 °C)", hoverinfo="skip",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(np.where(y_true <= -0.5, y_true, -0.5)) + [-0.5] * len(t),
            fill="toself", fillcolor="rgba(33,150,243,0.12)",
            line=dict(width=0), showlegend=(row == 1),
            name="La Niña (≤ −0.5 °C)", hoverinfo="skip",
        ), row=row, col=1)

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR,
                      line_width=1, row=row, col=1)

        # Actual
        fig.add_trace(go.Scatter(
            x=t, y=y_true,
            mode="lines", name="Actual Niño 3.4",
            line=dict(color=ACTUAL_CLR, width=2),
            hovertext=hover_actual, hoverinfo="text",
            showlegend=(row == 1),
        ), row=row, col=1)

        # Predicted
        fig.add_trace(go.Scatter(
            x=t, y=y_pred,
            mode="lines", name=f"Predicted (lead={lead}m)",
            line=dict(color=color, width=2, dash="dash"),
            hovertext=hover_pred, hoverinfo="text",
            showlegend=True,
        ), row=row, col=1)

        fig.update_yaxes(title_text="Niño 3.4 (°C)", row=row, col=1,
                         gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                         color=TEXT_COLOR)
        fig.update_xaxes(gridcolor=GRID_COLOR, color=TEXT_COLOR, row=row, col=1)

    fig.update_layout(
        title=dict(text="ENSO Niño 3.4 Forecast — Ridge Regression",
                   font=dict(size=20, color="white"), x=0.5),
        height=400 * n,
        **_plotly_dark_layout(),
    )

    path = f"{out_dir}/enso_predictions(ridge).html"
    pio.write_html(fig, path, include_plotlyjs="cdn")


def plot_performance_summary(results, out_dir):
    """
    Two side-by-side bar charts:
      - RMSE & MAE by lead time
      - Pearson correlation by lead time
    Saved to enso_performance.html
    """
    leads = [r["lead"] for r in results]
    rmses = [r["rmse"] for r in results]
    maes  = [r["mae"]  for r in results]
    corrs = [r["corr"] for r in results]
    colors = [_lead_color(l) for l in leads]
    x_labels = [f"{l}m" for l in leads]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Error Metrics by Lead Time",
                        "Correlation (Actual vs Predicted)"],
        horizontal_spacing=0.12,
    )

    # RMSE bars
    fig.add_trace(go.Bar(
        x=x_labels, y=rmses, name="RMSE",
        marker_color=[_lead_color(l) for l in leads],
        opacity=0.85,
        hovertemplate="<b>Lead %{x}</b><br>RMSE: %{y:.4f} °C<extra></extra>",
        text=[f"{v:.3f}" for v in rmses], textposition="outside",
        textfont=dict(color="white"),
    ), row=1, col=1)

    # MAE bars (grouped)
    fig.add_trace(go.Bar(
        x=x_labels, y=maes, name="MAE",
        marker_color="#9C27B0", opacity=0.7,
        hovertemplate="<b>Lead %{x}</b><br>MAE: %{y:.4f} °C<extra></extra>",
        text=[f"{v:.3f}" for v in maes], textposition="outside",
        textfont=dict(color="white"),
    ), row=1, col=1)

    # Correlation bars
    fig.add_trace(go.Bar(
        x=x_labels, y=corrs, name="Pearson r",
        marker_color=colors, opacity=0.85,
        hovertemplate="<b>Lead %{x}</b><br>Corr: %{y:.4f}<extra></extra>",
        text=[f"{v:.3f}" for v in corrs], textposition="outside",
        textfont=dict(color="white"),
    ), row=1, col=2)

    # Skill threshold line
    fig.add_hline(y=0.6, line_dash="dash", line_color="#FF9800", line_width=1.5,
                  annotation_text="r = 0.6 skill threshold",
                  annotation_font_color="#FF9800",
                  row=1, col=2)

    fig.update_layout(
        title=dict(text="Forecast Skill — Ridge Regression",
                   font=dict(size=18, color="white"), x=0.5),
        barmode="group",
        height=520,
        **_plotly_dark_layout(),
    )
    fig.update_yaxes(title_text="Error (°C)",  gridcolor=GRID_COLOR,
                     color=TEXT_COLOR, row=1, col=1)
    fig.update_yaxes(title_text="Pearson r",   gridcolor=GRID_COLOR,
                     color=TEXT_COLOR, range=[0, 1.15], row=1, col=2)
    fig.update_xaxes(gridcolor=GRID_COLOR, color=TEXT_COLOR)

    path = f"{out_dir}/enso_performance(ridge).html"
    pio.write_html(fig, path, include_plotlyjs="cdn")


def plot_error_timeseries(results_dict, out_dir):
    """
    Overlay of forecast error time-series for all lead times on one chart.
    Hover shows: time | error | lead
    Saved to enso_error_timeseries.html
    """
    fig = go.Figure()

    # ±0.3 °C band — use first lead's time axis as reference
    first = results_dict[sorted(results_dict.keys())[0]]
    t_ref = first["times_test"]
    fig.add_trace(go.Scatter(
        x=list(t_ref) + list(t_ref)[::-1],
        y=[0.3] * len(t_ref) + [-0.3] * len(t_ref),
        fill="toself", fillcolor="rgba(255,255,255,0.05)",
        line=dict(width=0), name="±0.3 °C band", hoverinfo="skip",
    ))

    for lead, r in sorted(results_dict.items()):
        error = r["y_true"] - r["y_pred"]
        color = _lead_color(lead)
        hover = [
            f"<b>Time:</b> {ts.strftime('%Y-%m')}<br>"
            f"<b>Lead:</b> {lead}m<br>"
            f"<b>Error:</b> {e:+.3f} °C"
            for ts, e in zip(r["times_test"], error)
        ]
        fig.add_trace(go.Scatter(
            x=r["times_test"], y=error,
            mode="lines", name=f"Lead {lead}m error",
            line=dict(color=color, width=1.8),
            hovertext=hover, hoverinfo="text",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.9)

    fig.update_layout(
        title=dict(text="Forecast Error Over Time by Lead",
                   font=dict(size=18, color="white"), x=0.5),
        xaxis_title="Time",
        yaxis_title="Error (actual − predicted)  °C",
        height=520,
        **_plotly_dark_layout(),
    )

    path = f"{out_dir}/enso_error_timeseries(ridge).html"
    pio.write_html(fig, path, include_plotlyjs="cdn")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(nino_path, soi_path, olr_path, sst_path,
                 lead_times=(1, 3, 6, 9, 12, 15),   # ← extended
                 alpha=1.0, train_frac=0.80,
                 out_dir="."):

    # 1. Load & clean
    df = load_and_merge(nino_path, soi_path, olr_path, sst_path)
    df = df.dropna(subset=["nino34"]).reset_index(drop=True)

    for col in ["soi", "olr", "sst_india"]:
        df[col] = df[col].interpolate(method="linear", limit=2)

    print(f"   Clean shape: {df.shape}\n")

    all_metrics  = []
    results_dict = {}

    print("Training Ridge Regression models\n")
    for lead in lead_times:
        X, y, times, feat_names = build_dataset(df, lead=lead)
        X_tr, X_te, y_tr, y_te, t_tr, t_te = time_split(X, y, times, train_frac)
        scaler, model = train_ridge(X_tr, y_tr, alpha)
        y_pred = model.predict(scaler.transform(X_te))
        m      = evaluate(y_te, y_pred, lead)
        all_metrics.append(m)

        results_dict[lead] = {
            "times_test": t_te.reset_index(drop=True),
            "y_true":     y_te,
            "y_pred":     y_pred,
            "metrics":    m,
        }
        print(f"  Lead {lead:>2}m | train n={len(y_tr):4d} | test n={len(y_te):4d} | "
              f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  Corr={m['corr']:.4f}")

    print_metrics_table(all_metrics)

    # 2. Interactive plots → HTML files
    print("Generating interactive plots")
    plot_predictions(results_dict,    out_dir)
    plot_performance_summary(all_metrics, out_dir)
    plot_error_timeseries(results_dict,   out_dir)

    print("\nCompleted")
    return results_dict, all_metrics


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )

    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "ridge")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_pipeline(
        nino_path  = os.path.join(DATA_DIR, "sstoi.indices.txt"),
        soi_path   = os.path.join(DATA_DIR, "soi.txt"),
        olr_path   = os.path.join(DATA_DIR, "olr.txt"),
        sst_path   = os.path.join(DATA_DIR, "sst_india.csv"),
        lead_times = [1, 3, 6, 9, 12, 15],
        alpha      = 1.0,
        train_frac = 0.80,
        out_dir    = OUTPUT_DIR,
    )
