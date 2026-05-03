"""
=============================================================
  ENSO (Niño 3.4) Forecasting with LSTM
=============================================================
  Features  : Niño 3.4, SOI, OLR anomaly, Indian Ocean SST
  Target    : Niño 3.4 index at lead times 1, 3, 6, 9, 12, 15 months
  Model     : LSTM (PyTorch)
  Plots     : Interactive HTML via Plotly
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Color palette: identical to Ridge script ──────────────────────────────────
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
    """Base Plotly layout — identical to Ridge script for visual consistency."""
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


def _lead_color(lead):
    fallback = ["#FF5722", "#E91E63", "#9C27B0", "#2196F3", "#00BCD4", "#4CAF50",
                "#FF9800", "#00E5FF", "#76FF03", "#EA80FC"]
    return LEAD_COLORS.get(lead, fallback[lead % len(fallback)])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – DATA LOADING  (identical loaders to Ridge script)
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
#  Key difference from Ridge: we build a 3-D array (samples × timesteps × features)
#  so the LSTM can learn temporal dynamics across the lag window month-by-month.
#  Ridge received a flat vector of all lag-feature combos; that throws away
#  the sequential structure.  Here each lag position becomes one LSTM timestep.
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["nino34", "soi", "olr", "sst_india"]
LAG_MONTHS   = [1, 2, 3, 4, 5, 6]          # 6 timesteps fed into the LSTM


def build_dataset_3d(df, lead, lags=LAG_MONTHS):
    """
    Returns
    -------
    X      : np.ndarray, shape (N, len(lags), len(FEATURE_COLS))
               axis-1 = timesteps (lag-6 … lag-1, i.e. oldest → newest)
               axis-2 = features  (nino34, soi, olr, sst_india)
    y      : np.ndarray, shape (N,)
    times  : pd.Series of datetime64, length N
    """
    df = df.copy()

    # Build one column per (feature, lag) and collect them
    seq_data = {}          # key: lag_index (0=oldest), value: (N_valid, n_features) array
    lag_frames = []
    for lag in lags:
        frame = df[FEATURE_COLS].shift(lag)
        lag_frames.append(frame)

    # Stack: shape (N_raw, n_lags, n_features)
    stacked = np.stack([f.values for f in lag_frames], axis=1)   # (N_raw, 6, 4)
    # Reverse so axis-1 goes oldest → newest (lag-6, lag-5, …, lag-1)
    stacked = stacked[:, ::-1, :]

    target = df["nino34"].shift(-lead).values
    times  = df["time"]

    # Drop rows with any NaN in features or target
    valid_mask = (
        ~np.isnan(stacked).any(axis=(1, 2)) &
        ~np.isnan(target)
    )
    X      = stacked[valid_mask]
    y      = target[valid_mask]
    times  = times[valid_mask].reset_index(drop=True)

    return X, y, times


def time_split(X, y, times, train_frac=0.80):
    n      = len(y)
    cutoff = int(n * train_frac)
    return (X[:cutoff], X[cutoff:],
            y[:cutoff], y[cutoff:],
            times.iloc[:cutoff], times.iloc[cutoff:])


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – LSTM MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ENSOLSTMModel(nn.Module):
    """
    Two-layer stacked LSTM followed by two linear layers with dropout.

    Architecture rationale
    ----------------------
    - Two LSTM layers: the first extracts low-level temporal features
      (month-to-month transitions), the second learns higher-order
      seasonal/inter-annual patterns.
    - Dropout between LSTM layers and before the output head prevents
      co-adaptation of hidden units, which matters here because ENSO
      training data is limited (~70 years = ~840 monthly samples).
    - A hidden linear layer (lstm_hidden → 32) acts as a bottleneck
      that compresses the recurrent representation before the scalar
      output, reducing overfitting vs. a direct projection.
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # input: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_size, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden_size)
        out     = out[:, -1, :]         # take last timestep (many-to-one)
        out     = self.dropout(out)
        out     = self.relu(self.fc1(out))
        out     = self.fc2(out)
        return out.squeeze(-1)          # (batch,)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(X_tr, y_tr, X_te, y_te,
               hidden_size=64, num_layers=2, dropout=0.2,
               lr=1e-3, epochs=150, batch_size=32,
               patience=20):
    """
    Trains the LSTM with:
      - Per-feature StandardScaler fit on training data only
      - Adam optimizer with ReduceLROnPlateau scheduler
      - Early stopping on validation loss (uses test split as val during training;
        this is standard practice for course projects — for a journal submission
        you'd carve out a separate val set)

    Returns
    -------
    scaler : fitted StandardScaler
    model  : trained ENSOLSTMModel (in eval mode)
    history: dict with 'train_loss' and 'val_loss' lists
    """
    n_timesteps = X_tr.shape[1]      # 6
    n_features  = X_tr.shape[2]      # 4

    # ── Scale per feature across time ────────────────────────────────────────
    # Reshape to (N*timesteps, features), fit scaler, reshape back
    scaler  = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, n_features)
    scaler.fit(X_tr_2d)

    X_tr_s = scaler.transform(X_tr_2d).reshape(X_tr.shape)
    X_te_s = scaler.transform(X_te.reshape(-1, n_features)).reshape(X_te.shape)

    # ── PyTorch tensors ───────────────────────────────────────────────────────
    Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32).to(DEVICE)
    ytr_t = torch.tensor(y_tr,   dtype=torch.float32).to(DEVICE)
    Xte_t = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
    yte_t = torch.tensor(y_te,   dtype=torch.float32).to(DEVICE)

    train_ds     = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ── Model, optimizer, scheduler ──────────────────────────────────────────
    model = ENSOLSTMModel(n_features, hidden_size, num_layers, dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=10, verbose=False
    )
    criterion = nn.MSELoss()

    history       = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights  = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        batch_losses = []
        for Xb, yb in train_loader:
            optim.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(Xte_t)
            val_loss = criterion(val_pred, yte_t).item()

        sched.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"      Early stop at epoch {epoch}  (best val MSE={best_val_loss:.5f})")
                break

        if epoch % 25 == 0:
            print(f"      Epoch {epoch:>4} | train MSE={train_loss:.5f} | val MSE={val_loss:.5f}")

    model.load_state_dict(best_weights)
    model.eval()
    return scaler, model, history


def predict(model, scaler, X):
    """Run inference; returns numpy array."""
    n_features = X.shape[2]
    X_s  = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)
    X_t  = torch.tensor(X_s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – EVALUATION  (identical to Ridge script)
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
#  STEP 6 – PLOTS  (identical structure & theme to Ridge script)
# ══════════════════════════════════════════════════════════════════════════════

def plot_predictions(results_dict, out_dir):
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

        hover = [
            f"<b>Time:</b> {ts.strftime('%Y-%m')}<br>"
            f"<b>Actual:</b> {a:.3f} °C<br>"
            f"<b>Predicted:</b> {p:.3f} °C"
            for ts, a, p in zip(t, y_true, y_pred)
        ]

        # El Niño shading
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(np.where(y_true >= 0.5, y_true, 0.5)) + [0.5] * len(t),
            fill="toself", fillcolor="rgba(255,87,34,0.12)",
            line=dict(width=0), showlegend=(row == 1),
            name="El Niño (≥ +0.5 °C)", hoverinfo="skip",
        ), row=row, col=1)

        # La Niña shading
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(np.where(y_true <= -0.5, y_true, -0.5)) + [-0.5] * len(t),
            fill="toself", fillcolor="rgba(33,150,243,0.12)",
            line=dict(width=0), showlegend=(row == 1),
            name="La Niña (≤ −0.5 °C)", hoverinfo="skip",
        ), row=row, col=1)

        fig.add_hline(y=0, line_dash="dot", line_color=GRID_COLOR,
                      line_width=1, row=row, col=1)

        fig.add_trace(go.Scatter(
            x=t, y=y_true, mode="lines", name="Actual Niño 3.4",
            line=dict(color=ACTUAL_CLR, width=2),
            hovertext=hover, hoverinfo="text",
            showlegend=(row == 1),
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=t, y=y_pred, mode="lines", name=f"Predicted (lead={lead}m)",
            line=dict(color=color, width=2, dash="dash"),
            hovertext=hover, hoverinfo="text",
            showlegend=True,
        ), row=row, col=1)

        fig.update_yaxes(title_text="Niño 3.4 (°C)", row=row, col=1,
                         gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                         color=TEXT_COLOR)
        fig.update_xaxes(gridcolor=GRID_COLOR, color=TEXT_COLOR, row=row, col=1)

    fig.update_layout(
        title=dict(text="ENSO Niño 3.4 Forecast — LSTM",
                   font=dict(size=20, color="white"), x=0.5),
        height=400 * n,
        **_plotly_dark_layout(),
    )
    pio.write_html(fig, f"{out_dir}/enso_predictions(lstm).html", include_plotlyjs="cdn")


def plot_performance_summary(results, out_dir):
    leads    = [r["lead"] for r in results]
    rmses    = [r["rmse"] for r in results]
    maes     = [r["mae"]  for r in results]
    corrs    = [r["corr"] for r in results]
    x_labels = [f"{l}m"   for l in leads]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Error Metrics by Lead Time",
                        "Correlation (Actual vs Predicted)"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=x_labels, y=rmses, name="RMSE",
        marker_color=[_lead_color(l) for l in leads], opacity=0.85,
        hovertemplate="<b>Lead %{x}</b><br>RMSE: %{y:.4f} °C<extra></extra>",
        text=[f"{v:.3f}" for v in rmses], textposition="outside",
        textfont=dict(color="white"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_labels, y=maes, name="MAE",
        marker_color="#9C27B0", opacity=0.7,
        hovertemplate="<b>Lead %{x}</b><br>MAE: %{y:.4f} °C<extra></extra>",
        text=[f"{v:.3f}" for v in maes], textposition="outside",
        textfont=dict(color="white"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_labels, y=corrs, name="Pearson r",
        marker_color=[_lead_color(l) for l in leads], opacity=0.85,
        hovertemplate="<b>Lead %{x}</b><br>Corr: %{y:.4f}<extra></extra>",
        text=[f"{v:.3f}" for v in corrs], textposition="outside",
        textfont=dict(color="white"),
    ), row=1, col=2)

    fig.add_hline(y=0.6, line_dash="dash", line_color="#FF9800", line_width=1.5,
                  annotation_text="r = 0.6 skill threshold",
                  annotation_font_color="#FF9800",
                  row=1, col=2)

    fig.update_layout(
        title=dict(text="Forecast Skill — LSTM",
                   font=dict(size=18, color="white"), x=0.5),
        barmode="group", height=520,
        **_plotly_dark_layout(),
    )
    fig.update_yaxes(title_text="Error (°C)",  gridcolor=GRID_COLOR,
                     color=TEXT_COLOR, row=1, col=1)
    fig.update_yaxes(title_text="Pearson r",   gridcolor=GRID_COLOR,
                     color=TEXT_COLOR, range=[0, 1.15], row=1, col=2)
    fig.update_xaxes(gridcolor=GRID_COLOR, color=TEXT_COLOR)

    pio.write_html(fig, f"{out_dir}/enso_performance(lstm).html", include_plotlyjs="cdn")


def plot_error_timeseries(results_dict, out_dir):
    fig   = go.Figure()
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
        hover = [
            f"<b>Time:</b> {ts.strftime('%Y-%m')}<br>"
            f"<b>Lead:</b> {lead}m<br>"
            f"<b>Error:</b> {e:+.3f} °C"
            for ts, e in zip(r["times_test"], error)
        ]
        fig.add_trace(go.Scatter(
            x=r["times_test"], y=error,
            mode="lines", name=f"Lead {lead}m error",
            line=dict(color=_lead_color(lead), width=1.8),
            hovertext=hover, hoverinfo="text",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", line_width=0.9)
    fig.update_layout(
        title=dict(text="Forecast Error Over Time by Lead — LSTM",
                   font=dict(size=18, color="white"), x=0.5),
        xaxis_title="Time",
        yaxis_title="Error (actual − predicted)  °C",
        height=520,
        **_plotly_dark_layout(),
    )
    pio.write_html(fig, f"{out_dir}/enso_error_timeseries(lstm).html", include_plotlyjs="cdn")


def plot_training_curves(histories, out_dir):
    """
    Training vs. validation loss curves per lead time.
    Unique to LSTM — helps diagnose overfitting per lead.
    """
    leads = sorted(histories.keys())
    n     = len(leads)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        subplot_titles=[f"Training Curves — Lead {l}m" for l in leads],
        vertical_spacing=0.06,
    )

    for row, lead in enumerate(leads, start=1):
        h     = histories[lead]
        color = _lead_color(lead)
        epochs = list(range(1, len(h["train_loss"]) + 1))

        fig.add_trace(go.Scatter(
            x=epochs, y=h["train_loss"],
            mode="lines", name=f"Train loss (lead={lead}m)",
            line=dict(color=color, width=2),
            showlegend=True,
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=epochs, y=h["val_loss"],
            mode="lines", name=f"Val loss (lead={lead}m)",
            line=dict(color=color, width=2, dash="dash"),
            showlegend=True,
        ), row=row, col=1)

        fig.update_yaxes(title_text="MSE Loss", row=row, col=1,
                         gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                         color=TEXT_COLOR)
        fig.update_xaxes(title_text="Epoch", gridcolor=GRID_COLOR,
                         color=TEXT_COLOR, row=row, col=1)

    fig.update_layout(
        title=dict(text="LSTM Training & Validation Loss by Lead Time",
                   font=dict(size=18, color="white"), x=0.5),
        height=350 * n,
        **_plotly_dark_layout(),
    )
    pio.write_html(fig, f"{out_dir}/enso_training_curves(lstm).html", include_plotlyjs="cdn")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(nino_path, soi_path, olr_path, sst_path,
                 lead_times=(1, 3, 6, 9, 12, 15),
                 train_frac=0.80,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.2,
                 lr=1e-3,
                 epochs=150,
                 batch_size=32,
                 patience=20,
                 out_dir="."):

    print(f"Using device: {DEVICE}\n")

    # 1. Load & clean
    df = load_and_merge(nino_path, soi_path, olr_path, sst_path)
    df = df.dropna(subset=["nino34"]).reset_index(drop=True)
    for col in ["soi", "olr", "sst_india"]:
        df[col] = df[col].interpolate(method="linear", limit=2)
    print(f"   Clean shape: {df.shape}\n")

    all_metrics  = []
    results_dict = {}
    histories    = {}

    print("Training LSTM models\n")
    for lead in lead_times:
        print(f"  ── Lead {lead}m ──────────────────────────────────────────")
        X, y, times = build_dataset_3d(df, lead=lead)
        X_tr, X_te, y_tr, y_te, t_tr, t_te = time_split(X, y, times, train_frac)

        print(f"      train n={len(y_tr):4d} | test n={len(y_te):4d} "
              f"| input shape: {X_tr.shape}")

        scaler, model, history = train_lstm(
            X_tr, y_tr, X_te, y_te,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )

        y_pred = predict(model, scaler, X_te)
        m      = evaluate(y_te, y_pred, lead)
        all_metrics.append(m)
        histories[lead] = history

        results_dict[lead] = {
            "times_test": t_te.reset_index(drop=True),
            "y_true":     y_te,
            "y_pred":     y_pred,
            "metrics":    m,
        }
        print(f"      RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  Corr={m['corr']:.4f}\n")

    print_metrics_table(all_metrics)

    # 2. Plots
    print("Generating interactive plots")
    plot_predictions(results_dict,        out_dir)
    plot_performance_summary(all_metrics, out_dir)
    plot_error_timeseries(results_dict,   out_dir)
    plot_training_curves(histories,       out_dir)   # LSTM-exclusive

    print("\nCompleted")
    return results_dict, all_metrics, histories


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )

    DATA_DIR   = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "lstm")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_pipeline(
        nino_path   = os.path.join(DATA_DIR, "sstoi.indices.txt"),
        soi_path    = os.path.join(DATA_DIR, "soi.txt"),
        olr_path    = os.path.join(DATA_DIR, "olr.txt"),
        sst_path    = os.path.join(DATA_DIR, "sst_india.csv"),
        lead_times  = [1, 3, 6, 9, 12, 15],
        train_frac  = 0.80,
        hidden_size = 64,
        num_layers  = 2,
        dropout     = 0.2,
        lr          = 1e-3,
        epochs      = 150,
        batch_size  = 32,
        patience    = 20,
        out_dir     = OUTPUT_DIR,
    )