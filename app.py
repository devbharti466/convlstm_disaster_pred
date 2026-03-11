# ============================================================
# 🌍 CONVLSTM DISASTER PREDICTION — FULL WEB APPLICATION
# ============================================================

import os
import random
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DisasterAI — Predict. Prepare. Protect.",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# FULL CUSTOM CSS — WEBAPP LOOK
# ============================================================
st.markdown("""
<style>
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark themed navbar */
    .navbar {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1rem 2rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    }
    .navbar-brand {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .navbar-brand span {
        color: #f39c12;
    }
    .navbar-links {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .hero p {
        font-size: 1.15rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* Glass card */
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Metric boxes */
    .metric-box {
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-box:hover {
        transform: translateY(-3px) scale(1.02);
    }
    .metric-box .value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.2);
    }
    .metric-box .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.85;
    }
    .metric-box .icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .bg-accuracy { background: linear-gradient(135deg, #667eea, #764ba2); }
    .bg-precision { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .bg-recall { background: linear-gradient(135deg, #F2994A, #F2C94C); }
    .bg-f1 { background: linear-gradient(135deg, #ee0979, #ff6a00); }
    .bg-auroc { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
    .bg-logloss { background: linear-gradient(135deg, #c31432, #240b36); }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .status-success {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .status-warning {
        background: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }

    /* Section header */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }

    /* Animated divider */
    .animated-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #ee0979, #F2994A, #11998e, #667eea);
        background-size: 300% 100%;
        animation: gradient-shift 5s ease infinite;
        border-radius: 2px;
        margin: 2rem 0;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Upload area */
    .upload-area {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e8ecf1, #b8c6db);
    }

    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div {
        color: #f39c12 !important;
    }

    /* Progress steps */
    .step-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    .step {
        background: rgba(102, 126, 234, 0.1);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        min-width: 150px;
        transition: all 0.3s ease;
    }
    .step.active {
        background: #667eea;
        color: white;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    .step .step-num {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .step .step-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Prediction card */
    .pred-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }

    /* Footer */
    .app-footer {
        background: linear-gradient(135deg, #0f0c29, #302b63);
        color: rgba(255,255,255,0.7);
        text-align: center;
        padding: 1.5rem;
        border-radius: 20px 20px 0 0;
        margin-top: 3rem;
        font-size: 0.85rem;
    }
    .app-footer a {
        color: #f39c12;
        text-decoration: none;
    }

    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: linear-gradient(135deg, #f5f7fa, #e4e8f0);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DEFINITIONS
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        dev = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=dev),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=dev),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(e, tuple) for e in kernel_size))):
            raise ValueError("`kernel_size` must be tuple or list of tuples")
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = [self.cell_list[i].init_hidden(b, (h, w)) for i in range(self.num_layers)]
        layer_output_list, last_state_list = [], []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h_cur, c_cur = hidden_state[layer_idx]
            output_inner = []
            for t in range(input_tensor.size(1)):
                h_cur, c_cur = self.cell_list[layer_idx](cur_layer_input[:, t], [h_cur, c_cur])
                output_inner.append(h_cur)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_cur, c_cur])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list


class ConvLSTMClassifier(nn.Module):
    def __init__(self, input_dim, patch_size, dropout=0.3):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=[64, 64, 64],
                                  kernel_size=(3, 3), num_layers=3,
                                  batch_first=True, bias=True, return_all_layers=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * patch_size * patch_size, 1)

    def forward(self, x):
        _, last_states = self.convlstm(x)
        h_last = last_states[-1][0]
        z = self.dropout(h_last)
        z = z.flatten(start_dim=1)
        logits = self.fc(z).squeeze(1)
        return torch.sigmoid(logits)


class DisasterPatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.eval() if optimizer is None else model.train()
    total_loss = 0.0
    all_probs, all_true = [], []
    ctx = torch.no_grad() if optimizer is None else torch.enable_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if optimizer:
                optimizer.zero_grad()
            probs = model(xb)
            loss = criterion(probs, yb)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            all_probs.append(probs.detach().cpu().numpy())
            all_true.append(yb.detach().cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, (all_probs >= 0.5).astype(int))
    return avg_loss, acc, all_true, all_probs

# ============================================================
# DATA PIPELINE
# ============================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_path, grid_size, patch_size, seq_len, pred_horizon):
    df = pd.read_excel(file_path)
    fully_null_cols = [c for c in df.columns if df[c].isna().all()]
    if fully_null_cols:
        df = df.drop(columns=fully_null_cols)

    if "Total Damage, Adjusted ('000 US$)" in df.columns:
        DAMAGE_COL = "Total Damage, Adjusted ('000 US$)"
    elif "Total Damage ('000 US$)" in df.columns:
        DAMAGE_COL = "Total Damage ('000 US$)"
    else:
        df["DamageUsed"] = 0.0
        DAMAGE_COL = "DamageUsed"

    if "Historic_Encoded" not in df.columns:
        df["Historic_Encoded"] = 0.0

    required_cols = ["Latitude", "Longitude", "Start Year", "Start Month",
                     "Total Deaths", "Total Affected", "CPI", DAMAGE_COL, "Historic_Encoded"]
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude", "Start Year", "Start Month"])
    df["Start Year"] = df["Start Year"].astype(int)
    df["Start Month"] = df["Start Month"].astype(int)
    df = df[(df["Start Month"] >= 1) & (df["Start Month"] <= 12)]
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    df["Date"] = pd.to_datetime(dict(year=df["Start Year"], month=df["Start Month"], day=1), errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Total Deaths"] = df["Total Deaths"].fillna(0)
    df["Total Affected"] = df["Total Affected"].fillna(0)
    df[DAMAGE_COL] = df[DAMAGE_COL].fillna(0)
    df["CPI"] = df["CPI"].fillna(df["CPI"].median())
    df["Historic_Encoded"] = df["Historic_Encoded"].fillna(0)
    df["DisasterOccurrence"] = ((df["Total Deaths"] > 0) | (df["Total Affected"] > 0)).astype(int)

    lat_min, lat_max = np.floor(df["Latitude"].min()), np.ceil(df["Latitude"].max())
    lon_min, lon_max = np.floor(df["Longitude"].min()), np.ceil(df["Longitude"].max())
    df["row"] = ((df["Latitude"] - lat_min) / grid_size).astype(int)
    df["col"] = ((df["Longitude"] - lon_min) / grid_size).astype(int)
    H = int(np.floor((lat_max - lat_min) / grid_size)) + 1
    W = int(np.floor((lon_max - lon_min) / grid_size)) + 1

    monthly_cell = df.groupby(["Date", "row", "col"], as_index=False).agg(
        event_count=("DisasterOccurrence", "size"), total_deaths=("Total Deaths", "sum"),
        total_affected=("Total Affected", "sum"), total_damage=(DAMAGE_COL, "sum"),
        cpi_mean=("CPI", "mean"), historic_mean=("Historic_Encoded", "mean"),
        disaster_occurrence=("DisasterOccurrence", "max"),
    )
    for col_name, src in [("event_count_log", "event_count"), ("deaths_log", "total_deaths"),
                           ("affected_log", "total_affected"), ("damage_log", "total_damage")]:
        monthly_cell[col_name] = np.log1p(monthly_cell[src])

    all_months = pd.date_range(start=monthly_cell["Date"].min(), end=monthly_cell["Date"].max(), freq="MS")
    T = len(all_months)
    month_to_idx = {d: i for i, d in enumerate(all_months)}
    C = 6

    X_monthly = np.zeros((T, H, W, C), dtype=np.float32)
    y_monthly = np.zeros((T, H, W), dtype=np.float32)
    for _, r in monthly_cell.iterrows():
        t, rr, cc = month_to_idx[r["Date"]], int(r["row"]), int(r["col"])
        X_monthly[t, rr, cc] = [r["event_count_log"], r["deaths_log"], r["affected_log"],
                                  r["damage_log"], r["cpi_mean"], r["historic_mean"]]
        y_monthly[t, rr, cc] = r["disaster_occurrence"]

    active_mask = X_monthly[..., 0].sum(axis=0) > 0
    active_cells = np.argwhere(active_mask)

    pad = patch_size // 2
    X_padded = np.pad(X_monthly, ((0,0),(pad,pad),(pad,pad),(0,0)), mode="constant")
    X_list, y_list, tmi = [], [], []
    for t_end in range(seq_len - 1, T - pred_horizon):
        label_t = t_end + pred_horizon
        seq_start = t_end - seq_len + 1
        for rc in active_cells:
            rr, cc = int(rc[0]), int(rc[1])
            patch = X_padded[seq_start:t_end+1, rr:rr+patch_size, cc:cc+patch_size, :]
            X_list.append(np.transpose(patch, (0, 3, 1, 2)))
            y_list.append(y_monthly[label_t, rr, cc])
            tmi.append(label_t)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    tmi = np.array(tmi, dtype=np.int32)

    um = np.array(sorted(np.unique(tmi)))
    n = len(um)
    tc = max(1, int(0.70 * n))
    vc = max(tc + 1, int(0.85 * n))
    if vc >= n: vc = n - 1

    X_train, y_train = X[np.isin(tmi, um[:tc])], y[np.isin(tmi, um[:tc])]
    X_val, y_val = X[np.isin(tmi, um[tc:vc])], y[np.isin(tmi, um[tc:vc])]
    X_test, y_test = X[np.isin(tmi, um[vc:])], y[np.isin(tmi, um[vc:])]

    mean = X_train.mean(axis=(0,1,3,4), keepdims=True)
    std = X_train.std(axis=(0,1,3,4), keepdims=True) + 1e-6
    X_train, X_val, X_test = (X_train-mean)/std, (X_val-mean)/std, (X_test-mean)/std

    active_lats = lat_min + active_cells[:, 0] * grid_size + grid_size / 2
    active_lons = lon_min + active_cells[:, 1] * grid_size + grid_size / 2

    return (df, X_train, y_train, X_val, y_val, X_test, y_test, C,
            active_cells, active_lats, active_lons, lat_min, lon_min,
            grid_size, all_months, H, W, y_monthly)


@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train, X_val, y_val, C, patch_size, dropout, lr, epochs, batch_size):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DisasterPatchDataset(X_train, y_train)
    val_ds = DisasterPatchDataset(X_val, y_val)

    y_int = y_train.astype(int)
    cc = np.bincount(y_int, minlength=2)
    if cc.min() > 0:
        sw = (1.0 / cc)[y_int]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvLSTMClassifier(input_dim=C, patch_size=patch_size, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_vl = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va, _, _ = run_epoch(model, val_loader, criterion, device, None)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, device

# ============================================================
# NAVBAR
# ============================================================
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">🌪️ Disaster<span>AI</span></div>
    <div class="navbar-links">ConvLSTM Spatio-Temporal Prediction Engine</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — CONTROL PANEL
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🧬 Model Config")

    grid_size = st.slider("🌐 Grid Size (°)", 0.5, 5.0, 1.0, 0.5)
    patch_size = st.selectbox("📐 Patch Size", [3, 5, 7], index=0)
    seq_len = st.slider("📅 Input Months", 3, 24, 12)
    pred_horizon = st.slider("🔮 Predict Ahead (months)", 1, 12, 6)

    st.markdown("---")
    st.markdown("### ⚡ Training")

    epochs = st.slider("🔄 Epochs", 2, 30, 8)
    batch_size = st.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
    lr = st.select_slider("📈 Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    dropout = st.slider("💧 Dropout", 0.0, 0.7, 0.3, 0.05)

    st.markdown("---")
    st.markdown("### 🎯 Inference")
    threshold = st.slider("⚖️ Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem;'>"
        "v2.0 · PyTorch · Streamlit"
        "</div>", unsafe_allow_html=True
    )

# ============================================================
# LANDING PAGE (No file uploaded)
# ============================================================
if uploaded_file is None:
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🌍 Predict Natural Disasters Before They Strike</h1>
        <p>Upload your spatio-temporal dataset and let our ConvLSTM deep learning engine<br>
        analyze patterns across space and time to forecast future disasters.</p>
    </div>
    """, unsafe_allow_html=True)

    # Steps
    st.markdown("""
    <div class="step-container">
        <div class="step active">
            <div class="step-num">01</div>
            <div class="step-label">Upload Data</div>
        </div>
        <div class="step">
            <div class="step-num">02</div>
            <div class="step-label">Auto Process</div>
        </div>
        <div class="step">
            <div class="step-num">03</div>
            <div class="step-label">Train Model</div>
        </div>
        <div class="step">
            <div class="step-num">04</div>
            <div class="step-label">View Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="pred-card">
            <h3>🧠 Deep Learning</h3>
            <p>3-layer ConvLSTM network captures complex spatio-temporal dependencies in disaster data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="pred-card">
            <h3>🗺️ Spatial Analysis</h3>
            <p>Geographic gridding with configurable resolution creates local spatial patches for prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="pred-card">
            <h3>📊 Rich Analytics</h3>
            <p>Interactive charts, geo maps, ROC curves, confusion matrices, and downloadable reports.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Required columns
    with st.expander("📋 Required Dataset Columns", expanded=False):
        st.markdown("""
        | Column | Type | Description |
        |--------|------|-------------|
        | `Latitude` | Float | -90 to 90 |
        | `Longitude` | Float | -180 to 180 |
        | `Start Year` | Int | Year of event |
        | `Start Month` | Int | 1–12 |
        | `Total Deaths` | Numeric | Fatalities |
        | `Total Affected` | Numeric | People affected |
        | `CPI` | Float | Consumer Price Index |
        | `Total Damage ('000 US$)` | Numeric | Economic damage |
        | `Historic_Encoded` | Float | Optional |
        """)

    st.stop()


# ============================================================
# FILE UPLOADED → PROCESS + TRAIN + DISPLAY
# ============================================================
tmp_path = "/tmp/disaster_data.xlsx"
with open(tmp_path, "wb") as f:
    f.write(uploaded_file.getvalue())

# Progress steps
st.markdown("""
<div class="step-container">
    <div class="step active"><div class="step-num">✅</div><div class="step-label">Data Uploaded</div></div>
    <div class="step active"><div class="step-num">02</div><div class="step-label">Processing</div></div>
    <div class="step"><div class="step-num">03</div><div class="step-label">Training</div></div>
    <div class="step"><div class="step-num">04</div><div class="step-label">Results</div></div>
</div>
""", unsafe_allow_html=True)

progress_bar = st.progress(0, text="🔄 Loading and preprocessing data...")

with st.spinner(""):
    data = load_and_preprocess(tmp_path, grid_size, patch_size, seq_len, pred_horizon)
    (df, X_train, y_train, X_val, y_val, X_test, y_test, C,
     active_cells, active_lats, active_lons, lat_min, lon_min,
     grid_size_val, all_months, H, W, y_monthly) = data

progress_bar.progress(40, text="🧠 Training ConvLSTM model...")

with st.spinner(""):
    model, history, device = train_model(
        X_train, y_train, X_val, y_val, C, patch_size, dropout, lr, epochs, batch_size
    )

progress_bar.progress(80, text="📊 Evaluating on test set...")

# Update steps
st.markdown("""
<div class="step-container">
    <div class="step active"><div class="step-num">✅</div><div class="step-label">Data Ready</div></div>
    <div class="step active"><div class="step-num">✅</div><div class="step-label">Processed</div></div>
    <div class="step active"><div class="step-num">✅</div><div class="step-label">Trained</div></div>
    <div class="step active"><div class="step-num">✅</div><div class="step-label">Results</div></div>
</div>
""", unsafe_allow_html=True)

# Status
pos_rate = y_true.mean()
st.markdown(f"""
<div style="text-align:center; margin:1rem 0;">
    <span class="status-badge status-success">● MODEL READY</span>&nbsp;&nbsp;
    <span class="status-badge status-warning">● POSITIVE RATE: {pos_rate:.1%}</span>&nbsp;&nbsp;
    <span class="status-badge status-success">● {len(X_test):,} TEST SAMPLES</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "📈 Training", "🌍 Geo Intelligence", "🔬 Deep Analysis", "📥 Export"
])


# ━━━━━━━━━━━━━━━━━━━━ TAB 1: OVERVIEW ━━━━━━━━━━━━━━━━━━━━
with tab1:
    # Quick stats row
    st.markdown('<div class="section-header">📌 Dataset at a Glance</div>', unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("📄 Total Records", f"{len(df):,}")
    q2.metric("🔲 Grid Cells", f"{len(active_cells):,}")
    q3.metric("📅 Time Span", f"{len(all_months)} months")
    q4.metric("🏋️ Train Size", f"{len(X_train):,}")
    q5.metric("🧪 Test Size", f"{len(X_test):,}")

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    # Metric cards
    st.markdown('<div class="section-header">🏆 Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics_data = [
        ("🎯", "Accuracy", f"{test_acc:.3f}", "bg-accuracy"),
        ("✅", "Precision", f"{prec:.3f}", "bg-precision"),
        ("🔍", "Recall", f"{rec:.3f}", "bg-recall"),
        ("⚡", "F1 Score", f"{f1:.3f}", "bg-f1"),
        ("📊", "AUROC", f"{auroc:.3f}", "bg-auroc"),
        ("📉", "Log Loss", f"{ll:.3f}", "bg-logloss"),
    ]
    for col, (icon, label, value, bg) in zip([m1, m2, m3, m4, m5, m6], metrics_data):
        col.markdown(f"""
        <div class="metric-box {bg}">
            <div class="icon">{icon}</div>
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    # Charts row
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=["No Disaster", "Disaster"], y=["No Disaster", "Disaster"],
                           color_continuous_scale="Purples", text_auto=True)
        fig_cm.update_layout(height=420, template="plotly_dark",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    with ch2:
        st.markdown('<div class="section-header">📊 Probability Distribution</div>', unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=y_prob[y_true == 0], name="No Disaster",
                                         opacity=0.75, marker_color="#667eea", nbinsx=50))
        fig_dist.add_trace(go.Histogram(x=y_prob[y_true == 1], name="Disaster",
                                         opacity=0.75, marker_color="#ee0979", nbinsx=50))
        fig_dist.update_layout(barmode="overlay", height=420, template="plotly_dark",
                               xaxis_title="Predicted Probability", yaxis_title="Count",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#f39c12",
                           annotation_text=f"Threshold: {threshold}")
        st.plotly_chart(fig_dist, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━ TAB 2: TRAINING ━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<div class="section-header">📈 Training History</div>', unsafe_allow_html=True)

    ep = list(range(1, len(history["train_acc"]) + 1))
    fig_hist = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    fig_hist.add_trace(go.Scatter(x=ep, y=history["train_acc"], name="Train Acc",
                                   line=dict(color="#667eea", width=3)), row=1, col=1)
    fig_hist.add_trace(go.Scatter(x=ep, y=history["val_acc"], name="Val Acc",
                                   line=dict(color="#ee0979", width=3, dash="dash")), row=1, col=1)
    fig_hist.add_trace(go.Scatter(x=ep, y=history["train_loss"], name="Train Loss",
                                   line=dict(color="#667eea", width=3)), row=1, col=2)
    fig_hist.add_trace(go.Scatter(x=ep, y=history["val_loss"], name="Val Loss",
                                   line=dict(color="#ee0979", width=3, dash="dash")), row=1, col=2)
    fig_hist.update_layout(height=450, template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_hist.update_xaxes(title_text="Epoch")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    # Epoch table
    st.markdown('<div class="section-header">📋 Epoch Log</div>', unsafe_allow_html=True)
    epoch_df = pd.DataFrame({
        "Epoch": ep,
        "Train Loss": [round(v, 4) for v in history["train_loss"]],
        "Val Loss": [round(v, 4) for v in history["val_loss"]],
        "Train Acc": [round(v, 4) for v in history["train_acc"]],
        "Val Acc": [round(v, 4) for v in history["val_acc"]],
    })
    st.dataframe(epoch_df, use_container_width=True, hide_index=True)

    # Model info
    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏗️ Architecture</div>', unsafe_allow_html=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Params", f"{total_params:,}")
    a2.metric("Trainable", f"{trainable:,}")
    a3.metric("Layers", "3 × ConvLSTM")
    a4.metric("Device", str(device).upper())

    with st.expander("🔍 View Full Model"):
        st.code(str(model), language="text")


# ━━━━━━━━━━━━━━━━━━━━ TAB 3: GEO ━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<div class="section-header">🌍 Global Disaster Risk Map</div>', unsafe_allow_html=True)

    avg_disaster = y_monthly.mean(axis=0)
    map_data = []
    for idx, (rr, cc) in enumerate(active_cells):
        map_data.append({"lat": active_lats[idx], "lon": active_lons[idx], "risk": avg_disaster[rr, cc]})
    map_df = pd.DataFrame(map_data)

    fig_map = px.scatter_geo(map_df, lat="lat", lon="lon", color="risk", size="risk",
                              size_max=15, color_continuous_scale="Inferno",
                              projection="natural earth",
                              labels={"risk": "Risk Score"})
    fig_map.update_layout(height=600, template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)",
                           geo=dict(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)",
                                    landcolor="#1a1a2e", oceancolor="#16213e",
                                    showocean=True, showlakes=True),
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    # Timeline
    st.markdown('<div class="section-header">📅 Disaster Timeline</div>', unsafe_allow_html=True)
    monthly_counts = y_monthly.sum(axis=(1, 2))
    month_labels = [m.strftime("%Y-%m") for m in all_months]
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=month_labels, y=monthly_counts, fill="tozeroy",
                                 line=dict(color="#667eea", width=2),
                                 fillcolor="rgba(102,126,234,0.3)"))
    fig_tl.update_layout(height=350, template="plotly_dark",
                          xaxis_title="Month", yaxis_title="Active Disaster Cells",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_tl, use_container_width=True)

    # Top risk zones
    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔥 Top 10 High-Risk Zones</div>', unsafe_allow_html=True)
    top_risk = map_df.nlargest(10, "risk").reset_index(drop=True)
    top_risk.index = top_risk.index + 1
    top_risk.columns = ["Latitude", "Longitude", "Risk Score"]
    st.dataframe(top_risk, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━ TAB 4: ANALYSIS ━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<div class="section-header">🔬 Model Diagnostics</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("#### 📈 ROC Curve")
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"AUROC = {auroc:.3f}",
                                          line=dict(color="#667eea", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                          name="Random", line=dict(dash="dash", color="gray")))
            fig_roc.update_layout(height=400, template="plotly_dark",
                                   xaxis_title="FPR", yaxis_title="TPR",
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_roc, use_container_width=True)
        except:
            st.warning("Cannot compute ROC (single class).")

    with r2:
        st.markdown("#### 📉 Precision-Recall Curve")
        try:
            pr_p, pr_r, _ = precision_recall_curve(y_true, y_prob)
            ap = auc(pr_r, pr_p)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=pr_r, y=pr_p, mode="lines",
                                         name=f"AP = {ap:.3f}",
                                         line=dict(color="#ee0979", width=3)))
            fig_pr.update_layout(height=400, template="plotly_dark",
                                  xaxis_title="Recall", yaxis_title="Precision",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pr, use_container_width=True)
        except:
            st.warning("Cannot compute PR curve.")

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    # Threshold analysis
    st.markdown('<div class="section-header">🎯 Threshold Optimizer</div>', unsafe_allow_html=True)
    st.caption("Find the optimal decision threshold for your use case.")

    thresholds = np.arange(0.05, 1.0, 0.025)
    th_data = {"Threshold": [], "F1": [], "Precision": [], "Recall": []}
    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        th_data["Threshold"].append(th)
        th_data["F1"].append(f1_score(y_true, yp, zero_division=0))
        th_data["Precision"].append(precision_score(y_true, yp, zero_division=0))
        th_data["Recall"].append(recall_score(y_true, yp, zero_division=0))

    fig_th = go.Figure()
    fig_th.add_trace(go.Scatter(x=th_data["Threshold"], y=th_data["F1"],
                                 name="F1", line=dict(color="#2ecc71", width=3)))
    fig_th.add_trace(go.Scatter(x=th_data["Threshold"], y=th_data["Precision"],
                                 name="Precision", line=dict(color="#3498db", width=2)))
    fig_th.add_trace(go.Scatter(x=th_data["Threshold"], y=th_data["Recall"],
                                 name="Recall", line=dict(color="#e74c3c", width=2)))
    fig_th.add_vline(x=threshold, line_dash="dash", line_color="#f39c12",
                      annotation_text=f"Current: {threshold}")

    best_f1_th = th_data["Threshold"][np.argmax(th_data["F1"])]
    fig_th.add_vline(x=best_f1_th, line_dash="dot", line_color="#2ecc71",
                      annotation_text=f"Best F1: {best_f1_th:.2f}")

    fig_th.update_layout(height=420, template="plotly_dark",
                          xaxis_title="Threshold", yaxis_title="Score",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_th, use_container_width=True)

    # Prediction scatter
    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔮 Prediction Scatter</div>', unsafe_allow_html=True)

    scatter_df = pd.DataFrame({"Index": range(len(y_prob)), "Probability": y_prob,
                                "Actual": ["Disaster" if y == 1 else "No Disaster" for y in y_true]})
    fig_scatter = px.scatter(scatter_df, x="Index", y="Probability", color="Actual",
                              color_discrete_map={"Disaster": "#ee0979", "No Disaster": "#667eea"},
                              opacity=0.6)
    fig_scatter.add_hline(y=threshold, line_dash="dash", line_color="#f39c12")
    fig_scatter.update_layout(height=400, template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━ TAB 5: EXPORT ━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown('<div class="section-header">📥 Export Center</div>', unsafe_allow_html=True)

    results_df = pd.DataFrame({"y_true": y_true.astype(int), "y_prob": np.round(y_prob, 4), "y_pred": y_pred})

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("""
        <div class="pred-card">
            <h3>📄 Predictions</h3>
            <p>True labels, probabilities, and predicted classes for all test samples.</p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button("⬇️ Download Predictions", results_df.to_csv(index=False),
                           "convlstm_predictions.csv", "text/csv", use_container_width=True)

    with e2:
        metrics_export = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUROC", "Log Loss", "Threshold"],
            "Value": [round(v, 4) for v in [test_acc, prec, rec, f1, auroc, ll, threshold]]
        })
        st.markdown("""
        <div class="pred-card">
            <h3>📊 Metrics Report</h3>
            <p>All evaluation metrics from the test set in a clean CSV format.</p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button("⬇️ Download Metrics", metrics_export.to_csv(index=False),
                           "convlstm_metrics.csv", "text/csv", use_container_width=True)

    with e3:
        hist_df = pd.DataFrame(history)
        hist_df.insert(0, "Epoch", range(1, len(hist_df) + 1))
        st.markdown("""
        <div class="pred-card">
            <h3>📈 Training Log</h3>
            <p>Epoch-by-epoch training and validation loss & accuracy history.</p>
        </div>
        """, unsafe_allow_html=True)
        st.download_button("⬇️ Download History", hist_df.to_csv(index=False),
                           "training_history.csv", "text/csv", use_container_width=True)

    st.markdown('<div class="animated-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">👁️ Results Preview</div>', unsafe_allow_html=True)
    st.dataframe(results_df.head(200), use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="app-footer">
    <strong>🌪️ DisasterAI</strong> — ConvLSTM Spatio-Temporal Prediction Engine<br>
    Grid: {H}×{W} · Sequence: {seq_len} months · Horizon: {pred_horizon} months · 
    {sum(p.numel() for p in model.parameters()):,} parameters<br>
    <span style="opacity:0.5;">Built with PyTorch + Streamlit · 2024</span>
</div>
""", unsafe_allow_html=True)