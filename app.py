# ============================================================
# STREAMLIT FRONTEND FOR CONVLSTM DISASTER PREDICTION
# ============================================================

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="🌍 Disaster Prediction — ConvLSTM",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 0.3rem 0 0 0;
        font-size: 2rem;
    }
    .metric-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-orange {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    .metric-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .metric-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL DEFINITIONS (same as your training code)
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
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
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
        batch_first=False, bias=True, return_all_layers=False,
    ):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not (len(kernel_size) == len(hidden_dim) == num_layers):
            raise ValueError("Inconsistent list length.")
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
            cell_list.append(
                ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], self.bias)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError()
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list, last_state_list = [], []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h_cur, c_cur = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_cur, c_cur = self.cell_list[layer_idx](
                    cur_layer_input[:, t, :, :, :], [h_cur, c_cur]
                )
                output_inner.append(h_cur)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_cur, c_cur])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [
            self.cell_list[i].init_hidden(batch_size, image_size)
            for i in range(self.num_layers)
        ]

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (isinstance(kernel_size, list) and all(isinstance(e, tuple) for e in kernel_size))
        ): 
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMClassifier(nn.Module):
    def __init__(self, input_dim, patch_size, dropout=0.3):
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=[64, 64, 64],
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * patch_size * patch_size, 1)

    def forward(self, x):
        _, last_states = self.convlstm(x)
        h_last = last_states[-1][0]
        z = self.dropout(h_last)
        z = z.flatten(start_dim=1)
        logits = self.fc(z).squeeze(1)
        return torch.sigmoid(logits)

# ============================================================
# DATASET
# ============================================================
class DisasterPatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# EVALUATION HELPER
# ============================================================
def run_epoch(model, loader, criterion, device, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    all_probs, all_true = [], []
    with torch.no_grad() if optimizer is None else torch.enable_grad():
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
# DATA PIPELINE (cached)
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

    required_cols = [
        "Latitude", "Longitude", "Start Year", "Start Month",
        "Total Deaths", "Total Affected", "CPI", DAMAGE_COL, "Historic_Encoded",
    ]
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude", "Start Year", "Start Month"])
    df["Start Year"] = df["Start Year"].astype(int)
    df["Start Month"] = df["Start Month"].astype(int)
    df = df[(df["Start Month"] >= 1) & (df["Start Month"] <= 12)]
    df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    df["Date"] = pd.to_datetime(
        dict(year=df["Start Year"], month=df["Start Month"], day=1), errors="coerce"
    )
    df = df.dropna(subset=["Date"])

    df["Total Deaths"] = df["Total Deaths"].fillna(0)
    df["Total Affected"] = df["Total Affected"].fillna(0)
    df[DAMAGE_COL] = df[DAMAGE_COL].fillna(0)
    df["CPI"] = df["CPI"].fillna(df["CPI"].median())
    df["Historic_Encoded"] = df["Historic_Encoded"].fillna(0)
    df["DisasterOccurrence"] = (
        (df["Total Deaths"] > 0) | (df["Total Affected"] > 0)
    ).astype(int)

    lat_min, lat_max = np.floor(df["Latitude"].min()), np.ceil(df["Latitude"].max())
    lon_min, lon_max = np.floor(df["Longitude"].min()), np.ceil(df["Longitude"].max())
    df["row"] = ((df["Latitude"] - lat_min) / grid_size).astype(int)
    df["col"] = ((df["Longitude"] - lon_min) / grid_size).astype(int)
    H = int(np.floor((lat_max - lat_min) / grid_size)) + 1
    W = int(np.floor((lon_max - lon_min) / grid_size)) + 1

    monthly_cell = (
        df.groupby(["Date", "row", "col"], as_index=False)
        .agg(
            event_count=("DisasterOccurrence", "size"),
            total_deaths=("Total Deaths", "sum"),
            total_affected=("Total Affected", "sum"),
            total_damage=(DAMAGE_COL, "sum"),
            cpi_mean=("CPI", "mean"),
            historic_mean=("Historic_Encoded", "mean"),
            disaster_occurrence=("DisasterOccurrence", "max"),
        )
    )
    monthly_cell["event_count_log"] = np.log1p(monthly_cell["event_count"])
    monthly_cell["deaths_log"] = np.log1p(monthly_cell["total_deaths"])
    monthly_cell["affected_log"] = np.log1p(monthly_cell["total_affected"])
    monthly_cell["damage_log"] = np.log1p(monthly_cell["total_damage"])

    all_months = pd.date_range(
        start=monthly_cell["Date"].min(), end=monthly_cell["Date"].max(), freq="MS"
    )
    T = len(all_months)
    month_to_idx = {d: i for i, d in enumerate(all_months)}
    C = 6

    X_monthly = np.zeros((T, H, W, C), dtype=np.float32)
    y_monthly = np.zeros((T, H, W), dtype=np.float32)

    for _, r in monthly_cell.iterrows():
        t = month_to_idx[r["Date"]]
        rr, cc = int(r["row"]), int(r["col"])
        X_monthly[t, rr, cc, 0] = r["event_count_log"]
        X_monthly[t, rr, cc, 1] = r["deaths_log"]
        X_monthly[t, rr, cc, 2] = r["affected_log"]
        X_monthly[t, rr, cc, 3] = r["damage_log"]
        X_monthly[t, rr, cc, 4] = r["cpi_mean"]
        X_monthly[t, rr, cc, 5] = r["historic_mean"]
        y_monthly[t, rr, cc] = r["disaster_occurrence"]

    active_mask = X_monthly[..., 0].sum(axis=0) > 0
    active_cells = np.argwhere(active_mask)

    pad = patch_size // 2
    X_monthly_padded = np.pad(
        X_monthly, pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant"
    )

    X_list, y_list, tmi = [], [], []
    for t_end in range(seq_len - 1, T - pred_horizon):
        label_t = t_end + pred_horizon
        seq_start = t_end - seq_len + 1
        for rc in active_cells:
            rr, cc = int(rc[0]), int(rc[1])
            patch_seq = X_monthly_padded[
                seq_start : t_end + 1, rr : rr + patch_size, cc : cc + patch_size, :
            ]
            patch_seq = np.transpose(patch_seq, (0, 3, 1, 2))
            X_list.append(patch_seq)
            y_list.append(y_monthly[label_t, rr, cc])
            tmi.append(label_t)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    tmi = np.array(tmi, dtype=np.int32)

    unique_months_idx = np.array(sorted(np.unique(tmi)))
    n_tm = len(unique_months_idx)
    train_cut = max(1, int(0.70 * n_tm))
    val_cut = max(train_cut + 1, int(0.85 * n_tm))
    if val_cut >= n_tm:
        val_cut = n_tm - 1

    train_m = unique_months_idx[:train_cut]
    val_m = unique_months_idx[train_cut:val_cut]
    test_m = unique_months_idx[val_cut:]

    X_train, y_train = X[np.isin(tmi, train_m)], y[np.isin(tmi, train_m)]
    X_val, y_val = X[np.isin(tmi, val_m)], y[np.isin(tmi, val_m)]
    X_test, y_test = X[np.isin(tmi, test_m)], y[np.isin(tmi, test_m)]

    mean = X_train.mean(axis=(0, 1, 3, 4), keepdims=True)
    std = X_train.std(axis=(0, 1, 3, 4), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Build location mapping for active cells
    active_lats = lat_min + active_cells[:, 0] * grid_size + grid_size / 2
    active_lons = lon_min + active_cells[:, 1] * grid_size + grid_size / 2

    return (
        df, X_train, y_train, X_val, y_val, X_test, y_test,
        C, active_cells, active_lats, active_lons,
        lat_min, lon_min, grid_size, all_months, H, W, y_monthly,
    )

# ============================================================
# TRAINING (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def train_model(
    X_train, y_train, X_val, y_val, C, patch_size, dropout, lr, epochs, batch_size
):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DisasterPatchDataset(X_train, y_train)
    val_ds = DisasterPatchDataset(X_val, y_val)

    y_train_int = y_train.astype(int)
    class_counts = np.bincount(y_train_int, minlength=2)
    if class_counts.min() > 0:
        cw = 1.0 / class_counts
        sw = cw[y_train_int]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvLSTMClassifier(input_dim=C, patch_size=patch_size, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        tl, ta, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va, _, _ = run_epoch(model, val_loader, criterion, device, None)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, device

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image(
    "https://img.icons8.com/3d-fluency/94/earth-globe.png", width=80
)
st.sidebar.title("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset (.xlsx)", type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Hyperparameters")

grid_size = st.sidebar.slider("Grid Size (degrees)", 0.5, 5.0, 1.0, 0.5)
patch_size = st.sidebar.selectbox("Patch Size", [3, 5, 7], index=0)
seq_len = st.sidebar.slider("Sequence Length (months)", 3, 24, 12)
pred_horizon = st.sidebar.slider("Prediction Horizon (months)", 1, 12, 6)
epochs = st.sidebar.slider("Training Epochs", 2, 30, 8)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
lr = st.sidebar.select_slider(
    "Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3
)
dropout = st.sidebar.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
threshold = st.sidebar.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">🌍 Spatio-Temporal Disaster Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">ConvLSTM Deep Learning Model &nbsp;·&nbsp; '
    "Powered by PyTorch & Streamlit</p>",
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("👈 Upload your dataset from the sidebar to get started.")
    st.markdown("### 📋 Expected Dataset Columns")
    st.markdown(
        """
        | Column | Description |
        |--------|-------------|
        | `Latitude` | Event latitude (-90 to 90) |
        | `Longitude` | Event longitude (-180 to 180) |
        | `Start Year` | Year of the event |
        | `Start Month` | Month of the event (1–12) |
        | `Total Deaths` | Number of deaths |
        | `Total Affected` | Number of affected people |
        | `CPI` | Consumer Price Index |
        | `Total Damage ('000 US$)` | Economic damage |
        | `Historic_Encoded` | Historical encoding (optional) |
        """
    )
    st.stop()

# ============================================================
# SAVE UPLOADED FILE TEMPORARILY
# ============================================================
tmp_path = "/tmp/disaster_data.xlsx"
with open(tmp_path, "wb") as f:
    f.write(uploaded_file.getvalue())

# ============================================================
# PIPELINE
# ============================================================
with st.spinner("🔄 Loading and preprocessing data..."):
    (
        df, X_train, y_train, X_val, y_val, X_test, y_test,
        C, active_cells, active_lats, active_lons,
        lat_min, lon_min, grid_size_val, all_months, H, W, y_monthly,
    ) = load_and_preprocess(tmp_path, grid_size, patch_size, seq_len, pred_horizon)

with st.spinner("🧠 Training ConvLSTM model... This may take a few minutes."):
    model, history, device = train_model(
        X_train, y_train, X_val, y_val, C, patch_size, dropout, lr, epochs, batch_size
    )

# ============================================================
# TEST EVALUATION
# ============================================================
test_ds = DisasterPatchDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
criterion = nn.BCELoss()

test_loss, test_acc, y_true, y_prob = run_epoch(model, test_loader, criterion, device, None)
y_pred = (y_prob >= threshold).astype(int)

prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
try:
    auroc = roc_auc_score(y_true, y_prob)
except:
    auroc = float("nan")
try:
    ll = log_loss(y_true, y_prob, labels=[0, 1])
except:
    ll = float("nan")

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "📈 Training Curves", "🗺️ Geo Map", "🔬 Analysis", "📥 Downloads"]
)

# ────────────────────────── TAB 1 : DASHBOARD ──────────────────────────
with tab1:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("📌 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Active Grid Cells", f"{len(active_cells):,}")
    c3.metric("Train Samples", f"{len(X_train):,}")
    c4.metric("Test Samples", f"{len(X_test):,}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🏆 Test Set Performance")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.markdown(
        f'<div class="metric-card metric-blue"><h3>Accuracy</h3><h1>{test_acc:.3f}</h1></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card metric-green"><h3>Precision</h3><h1>{prec:.3f}</h1></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card metric-orange"><h3>Recall</h3><h1>{rec:.3f}</h1></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card"><h3>F1-Score</h3><h1>{f1:.3f}</h1></div>',
        unsafe_allow_html=True,
    )
    m5.markdown(
        f'<div class="metric-card metric-red"><h3>AUROC</h3><h1>{auroc:.3f}</h1></div>',
        unsafe_allow_html=True,
    )
    m6.markdown(
        f'<div class="metric-card metric-blue"><h3>Log Loss</h3><h1>{ll:.3f}</h1></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # CONFUSION MATRIX
    col_cm, col_dist = st.columns(2)

    with col_cm:
        st.subheader("🔲 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Disaster", "Disaster"],
            y=["No Disaster", "Disaster"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_dist:
        st.subheader("📊 Prediction Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=y_prob[y_true == 0], name="No Disaster", opacity=0.7,
                marker_color="#3498db", nbinsx=50,
            )
        )
        fig_dist.add_trace(
            go.Histogram(
                x=y_prob[y_true == 1], name="Disaster", opacity=0.7,
                marker_color="#e74c3c", nbinsx=50,
            )
        )
        fig_dist.update_layout(
            barmode="overlay", xaxis_title="Predicted Probability",
            yaxis_title="Count", height=400,
        )
        fig_dist.add_vline(x=threshold, line_dash="dash", line_color="green",
                           annotation_text=f"Threshold={threshold}")
        st.plotly_chart(fig_dist, use_container_width=True)


# ────────────────────────── TAB 2 : TRAINING CURVES ──────────────────────────
with tab2:
    st.subheader("📈 Training History")

    fig_hist = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy", "Loss"),
    )
    ep = list(range(1, len(history["train_acc"]) + 1))

    fig_hist.add_trace(
        go.Scatter(x=ep, y=history["train_acc"], name="Train Acc", line=dict(color="#2ecc71")), row=1, col=1
    )
    fig_hist.add_trace(
        go.Scatter(x=ep, y=history["val_acc"], name="Val Acc", line=dict(color="#e67e22", dash="dash")), row=1, col=1
    )
    fig_hist.add_trace(
        go.Scatter(x=ep, y=history["train_loss"], name="Train Loss", line=dict(color="#3498db")), row=1, col=2
    )
    fig_hist.add_trace(
        go.Scatter(x=ep, y=history["val_loss"], name="Val Loss", line=dict(color="#e74c3c", dash="dash")), row=1, col=2
    )
    fig_hist.update_layout(height=450, template="plotly_white")
    fig_hist.update_xaxes(title_text="Epoch")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Epoch-by-epoch table
    st.subheader("📋 Epoch Details")
    epoch_df = pd.DataFrame(
        {
            "Epoch": ep,
            "Train Loss": [f"{v:.4f}" for v in history["train_loss"]],
            "Val Loss": [f"{v:.4f}" for v in history["val_loss"]],
            "Train Acc": [f"{v:.4f}" for v in history["train_acc"]],
            "Val Acc": [f"{v:.4f}" for v in history["val_acc"]],
        }
    )
    st.dataframe(epoch_df, use_container_width=True, hide_index=True)


# ────────────────────────── TAB 3 : GEO MAP ──────────────────────────
with tab3:
    st.subheader("🗺️ Disaster Risk Heatmap")
    st.caption("Showing average disaster occurrence across all time steps for active grid cells.")

    # Average disaster probability per active cell across time
    avg_disaster = y_monthly.mean(axis=0)
    map_data = []
    for idx, (rr, cc) in enumerate(active_cells):
        map_data.append(
            {
                "lat": active_lats[idx],
                "lon": active_lons[idx],
                "risk": avg_disaster[rr, cc],
            }
        )
    map_df = pd.DataFrame(map_data)

    fig_map = px.scatter_geo(
        map_df,
        lat="lat",
        lon="lon",
        color="risk",
        size="risk",
        size_max=12,
        color_continuous_scale="YlOrRd",
        projection="natural earth",
        title="Average Disaster Risk by Grid Cell",
        labels={"risk": "Avg Risk"},
    )
    fig_map.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Monthly disaster count over time
    st.subheader("📅 Monthly Disaster Events Over Time")
    monthly_counts = y_monthly.sum(axis=(1, 2))
    month_labels = [m.strftime("%Y-%m") for m in all_months]
    fig_timeline = px.area(
        x=month_labels, y=monthly_counts,
        labels={"x": "Month", "y": "Active Disaster Cells"},
        title="Disaster Cell Count Over Time",
    )
    fig_timeline.update_layout(height=350)
    st.plotly_chart(fig_timeline, use_container_width=True)


# ────────────────────────── TAB 4 : ANALYSIS ──────────────────────────
with tab4:
    st.subheader("🔬 Detailed Model Analysis")

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curve")
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUROC={auroc:.3f}",
                           line=dict(color="#8e44ad", width=2))
            )
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                           line=dict(dash="dash", color="gray"))
            )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except:
            st.warning("Cannot compute ROC curve (single class in test set).")

    with col_pr:
        st.markdown("#### Precision-Recall Curve")
        try:
            pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
            ap = auc(pr_rec, pr_prec)
            fig_pr = go.Figure()
            fig_pr.add_trace(
                go.Scatter(x=pr_rec, y=pr_prec, mode="lines", name=f"AP={ap:.3f}",
                           line=dict(color="#e67e22", width=2))
            )
            fig_pr.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        except:
            st.warning("Cannot compute PR curve.")

    # Threshold Sensitivity
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🎯 Threshold Sensitivity Analysis")

    thresholds = np.arange(0.1, 1.0, 0.05)
    th_f1, th_prec, th_rec = [], [], []
    for th in thresholds:
        yp = (y_prob >= th).astype(int)
        th_f1.append(f1_score(y_true, yp, zero_division=0))
        th_prec.append(precision_score(y_true, yp, zero_division=0))
        th_rec.append(recall_score(y_true, yp, zero_division=0))

    fig_th = go.Figure()
    fig_th.add_trace(go.Scatter(x=thresholds, y=th_f1, name="F1", line=dict(color="#2ecc71")))
    fig_th.add_trace(go.Scatter(x=thresholds, y=th_prec, name="Precision", line=dict(color="#3498db")))
    fig_th.add_trace(go.Scatter(x=thresholds, y=th_rec, name="Recall", line=dict(color="#e74c3c")))
    fig_th.add_vline(x=threshold, line_dash="dash", line_color="gray",
                     annotation_text=f"Current={threshold}")
    fig_th.update_layout(
        xaxis_title="Threshold", yaxis_title="Score",
        height=400, template="plotly_white",
    )
    st.plotly_chart(fig_th, use_container_width=True)

    # Model Architecture
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("🏗️ Model Architecture")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    a1, a2, a3 = st.columns(3)
    a1.metric("Total Parameters", f"{total_params:,}")
    a2.metric("Trainable Parameters", f"{trainable_params:,}")
    a3.metric("Device", str(device))

    with st.expander("View Full Architecture"):
        st.code(str(model), language="text")


# ────────────────────────── TAB 5 : DOWNLOADS ──────────────────────────
with tab5:
    st.subheader("📥 Download Results")

    results_df = pd.DataFrame(
        {"y_true": y_true.astype(int), "y_prob": y_prob, "y_pred": y_pred}
    )

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📄 Download Predictions (CSV)",
            csv,
            "convlstm_predictions.csv",
            "text/csv",
        )

    with col_d2:
        metrics_df = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUROC", "Log Loss"],
                "Value": [test_acc, prec, rec, f1, auroc, ll],
            }
        )
        st.download_button(
            "📊 Download Metrics (CSV)",
            metrics_df.to_csv(index=False),
            "convlstm_metrics.csv",
            "text/csv",
        )

    with col_d3:
        hist_df = pd.DataFrame(history)
        st.download_button(
            "📈 Download Training History (CSV)",
            hist_df.to_csv(index=False),
            "training_history.csv",
            "text/csv",
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Preview")
    st.dataframe(results_df.head(100), use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🌪️ ConvLSTM Disaster Prediction Dashboard &nbsp;·&nbsp; "
    "Built with Streamlit + PyTorch &nbsp;·&nbsp; "
    f"Grid: {H}×{W} &nbsp;·&nbsp; Seq: {seq_len}m &nbsp;·&nbsp; Horizon: {pred_horizon}m"
    "</div>",
    unsafe_allow_html=True,
)