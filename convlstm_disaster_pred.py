# ============================================================
# FULL PYTORCH CONVLSTM CODE FOR SPATIO-TEMPORAL DISASTER PREDICTION
# ============================================================

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    auc
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ============================================================
# 1. REPRODUCIBILITY + DEVICE
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 2. CONFIG
# ============================================================
FILE_PATH = "/content/drive/MyDrive/Project IV Sem /improved_ordinal_regression_dataset_with_location.xlsx"

GRID_SIZE_DEG = 1.0      # 1° x 1° grid
PATCH_SIZE = 3           # 3x3 spatial patch
SEQ_LEN = 12             # 12 months input
PRED_HORIZON = 6         # 6 months ahead; change to 12 if needed
EPOCHS = 8
BATCH_SIZE = 32
LR = 1e-3
DROPOUT = 0.3


# ============================================================
# 3. CONVLSTM CELL
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

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
            bias=self.bias
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
            torch.zeros(batch_size, self.hidden_dim, height, width, device=dev)
        )


# ============================================================
# 4. MULTI-LAYER CONVLSTM
# ============================================================
class ConvLSTM(nn.Module):
    def __init__(self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False
    ):
        super(ConvLSTM, self).__init__()

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
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # expected input shape if batch_first=True: (B, T, C, H, W)
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError("Stateful ConvLSTM is not implemented.")
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_cur, c_cur = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h_cur, c_cur = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h_cur, c_cur]
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
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all(isinstance(elem, tuple) for elem in kernel_size)
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# ============================================================
# 5. CLASSIFIER WRAPPER
# ============================================================
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
            return_all_layers=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * patch_size * patch_size, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        _, last_states = self.convlstm(x)
        h_last = last_states[-1][0]           # (B, 64, H, W)
        z = self.dropout(h_last)
        z = z.flatten(start_dim=1)            # (B, 64*H*W)
        logits = self.fc(z).squeeze(1)        # (B,)
        probs = torch.sigmoid(logits)
        return probs


# ============================================================
# 6. LOAD DATA
# ============================================================
df = pd.read_excel(FILE_PATH)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop fully null columns
fully_null_cols = [c for c in df.columns if df[c].isna().all()]
if fully_null_cols:
    print("Dropping fully null columns:", fully_null_cols)
    df = df.drop(columns=fully_null_cols)

# Pick damage column robustly
if "Total Damage, Adjusted ('000 US$)" in df.columns:
    DAMAGE_COL = "Total Damage, Adjusted ('000 US$)"
elif "Total Damage ('000 US$)" in df.columns:
    DAMAGE_COL = "Total Damage ('000 US$)"
else:
    df["DamageUsed"] = 0.0
    DAMAGE_COL = "DamageUsed"

# Ensure Historic_Encoded exists
if "Historic_Encoded" not in df.columns:
    df["Historic_Encoded"] = 0.0

required_cols = [
    "Latitude",
    "Longitude",
    "Start Year",
    "Start Month",
    "Total Deaths",
    "Total Affected",
    "CPI",
    DAMAGE_COL,
    "Historic_Encoded"
]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

for c in required_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# ============================================================
# 7. BASIC CLEANING
# ============================================================
df = df.dropna(subset=["Latitude", "Longitude", "Start Year", "Start Month"])

df["Start Year"] = df["Start Year"].astype(int)
df["Start Month"] = df["Start Month"].astype(int)

df = df[(df["Start Month"] >= 1) & (df["Start Month"] <= 12)]
df = df[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]

df["Date"] = pd.to_datetime(
    dict(year=df["Start Year"], month=df["Start Month"], day=1),
    errors="coerce"
)
df = df.dropna(subset=["Date"]) 
df["Total Deaths"] = df["Total Deaths"].fillna(0)
df["Total Affected"] = df["Total Affected"].fillna(0)
df[DAMAGE_COL] = df[DAMAGE_COL].fillna(0)
df["CPI"] = df["CPI"].fillna(df["CPI"].median())
df["Historic_Encoded"] = df["Historic_Encoded"].fillna(0)

# Binary target
df["DisasterOccurrence"] = (
    (df["Total Deaths"] > 0) | (df["Total Affected"] > 0)
).astype(int)

print("Cleaned shape:", df.shape)
print("Positive rate at row level:", df["DisasterOccurrence"].mean())


# ============================================================
# 8. SPATIAL GRIDDING
# ============================================================
lat_min = np.floor(df["Latitude"].min())
lat_max = np.ceil(df["Latitude"].max())
lon_min = np.floor(df["Longitude"].min())
lon_max = np.ceil(df["Longitude"].max())

df["row"] = ((df["Latitude"] - lat_min) / GRID_SIZE_DEG).astype(int)
df["col"] = ((df["Longitude"] - lon_min) / GRID_SIZE_DEG).astype(int)

H = int(np.floor((lat_max - lat_min) / GRID_SIZE_DEG)) + 1
W = int(np.floor((lon_max - lon_min) / GRID_SIZE_DEG)) + 1

print(f"Grid shape: H={{H}}, W={{W}}")


# ============================================================
# 9. AGGREGATE TO MONTH-CELL LEVEL
# ============================================================
monthly_cell = (
    df.groupby(["Date", "row", "col"], as_index=False)
      .agg(
          event_count=("DisasterOccurrence", "size"),
          total_deaths=("Total Deaths", "sum"),
          total_affected=("Total Affected", "sum"),
          total_damage=(DAMAGE_COL, "sum"),
          cpi_mean=("CPI", "mean"),
          historic_mean=("Historic_Encoded", "mean"),
          disaster_occurrence=("DisasterOccurrence", "max")
      )
)

monthly_cell["event_count_log"] = np.log1p(monthly_cell["event_count"])
monthly_cell["deaths_log"] = np.log1p(monthly_cell["total_deaths"])
monthly_cell["affected_log"] = np.log1p(monthly_cell["total_affected"])
monthly_cell["damage_log"] = np.log1p(monthly_cell["total_damage"])

print("Monthly cell shape:", monthly_cell.shape)


# ============================================================
# 10. BUILD MONTHLY TENSORS
# ============================================================
all_months = pd.date_range(
    start=monthly_cell["Date"].min(),
    end=monthly_cell["Date"].max(),
    freq="MS"
)

T = len(all_months)
month_to_idx = {d: i for i, d in enumerate(all_months)}

channel_names = [
    "event_count_log",
    "deaths_log",
    "affected_log",
    "damage_log",
    "cpi_mean",
    "historic_mean"
]
C = len(channel_names)

X_monthly = np.zeros((T, H, W, C), dtype=np.float32)
y_monthly = np.zeros((T, H, W), dtype=np.float32)

for _, r in monthly_cell.iterrows():
    t = month_to_idx[r["Date"]]
    rr = int(r["row"])
    cc = int(r["col"])

    X_monthly[t, rr, cc, 0] = r["event_count_log"]
    X_monthly[t, rr, cc, 1] = r["deaths_log"]
    X_monthly[t, rr, cc, 2] = r["affected_log"]
    X_monthly[t, rr, cc, 3] = r["damage_log"]
    X_monthly[t, rr, cc, 4] = r["cpi_mean"]
    X_monthly[t, rr, cc, 5] = r["historic_mean"]

    y_monthly[t, rr, cc] = r["disaster_occurrence"]

print("Monthly tensor shape:", X_monthly.shape)


# ============================================================
# 11. ACTIVE CELLS
# ============================================================
active_mask = X_monthly[..., 0].sum(axis=0) > 0
active_cells = np.argwhere(active_mask)

print("Number of active cells:", len(active_cells))
if len(active_cells) == 0:
    raise ValueError("No active cells found. Try increasing GRID_SIZE_DEG.")


# ============================================================
# 12. CREATE LOCAL SPATIO-TEMPORAL SAMPLES
# ============================================================
pad = PATCH_SIZE // 2
X_monthly_padded = np.pad(
    X_monthly,
    pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)),
    mode="constant"
)

X_list = []
y_list = []
target_month_idx = []

# Input: months [t-SEQ_LEN+1, ..., t]
# Target: center cell at month t + PRED_HORIZON
for t_end in range(SEQ_LEN - 1, T - PRED_HORIZON):
    label_t = t_end + PRED_HORIZON
    seq_start = t_end - SEQ_LEN + 1
    seq_end = t_end + 1

    for rc in active_cells:
        rr, cc = int(rc[0]), int(rc[1])

        patch_seq = X_monthly_padded[
            seq_start:seq_end,
            rr:rr + PATCH_SIZE,
            cc:cc + PATCH_SIZE,
            :
        ]  # (SEQ_LEN, PATCH, PATCH, C)

        # PyTorch ConvLSTM expects (SEQ_LEN, C, PATCH, PATCH)
        patch_seq = np.transpose(patch_seq, (0, 3, 1, 2))

        label = y_monthly[label_t, rr, cc]

        X_list.append(patch_seq)
        y_list.append(label)
        target_month_idx.append(label_t)

X = np.array(X_list, dtype=np.float32)    # (N, T, C, H, W)
y = np.array(y_list, dtype=np.float32)    # (N,)
target_month_idx = np.array(target_month_idx, dtype=np.int32)

print("Sample tensor X shape:", X.shape)
print("Label y shape:", y.shape)

if len(X) == 0:
    raise ValueError("No samples created. Reduce SEQ_LEN or PRED_HORIZON.")

print("Positive rate at sample level:", y.mean())


# ============================================================
# 13. CHRONOLOGICAL SPLIT BY TARGET MONTH
# ============================================================
unique_months = np.array(sorted(np.unique(target_month_idx)))
n_target_months = len(unique_months)

print("Number of unique target months:", n_target_months)
if n_target_months < 3:
    raise ValueError("Too few target months for train/val/test split.")

train_cut = max(1, int(0.70 * n_target_months))
val_cut = max(train_cut + 1, int(0.85 * n_target_months))
if val_cut >= n_target_months:
    val_cut = n_target_months - 1

train_months = unique_months[:train_cut]
val_months = unique_months[train_cut:val_cut]
test_months = unique_months[val_cut:]

train_mask = np.isin(target_month_idx, train_months)
val_mask = np.isin(target_month_idx, val_months)
test_mask = np.isin(target_month_idx, test_months)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError("One split is empty. Reduce PRED_HORIZON or SEQ_LEN.")


# ============================================================
# 14. NORMALIZE USING TRAINING DATA ONLY
# ============================================================
# shape: (N, T, C, H, W)
mean = X_train.mean(axis=(0, 1, 3, 4), keepdims=True)   # (1,1,C,1,1)
std = X_train.std(axis=(0, 1, 3, 4), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# ============================================================
# 15. DATASET / DATALOADER
# ============================================================
class DisasterPatchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = DisasterPatchDataset(X_train, y_train)
val_ds = DisasterPatchDataset(X_val, y_val)
test_ds = DisasterPatchDataset(X_test, y_test)

# Balanced sampling instead of SMOTE on 5-D tensors
y_train_int = y_train.astype(int)
class_counts = np.bincount(y_train_int, minlength=2)
print("Train class counts:", class_counts)

if class_counts.min() > 0:
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train_int]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
else:
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 16. MODEL
# ============================================================
model = ConvLSTMClassifier(
    input_dim=C,
    patch_size=PATCH_SIZE,
    dropout=DROPOUT
).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)


# ============================================================
# 17. TRAIN / EVAL HELPER
# ============================================================
def run_epoch(model, loader, criterion, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    all_probs = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        probs = model(xb)
        loss = criterion(probs, yb)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_true.append(yb.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)

    avg_loss = total_loss / len(loader.dataset)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_true, preds)

    return avg_loss, acc, all_true, all_probs


# ============================================================
# 18. TRAINING LOOP
# ============================================================
best_val_loss = float("inf")
best_state = None

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

for epoch in range(EPOCHS):
    train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer=None)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {{epoch+1:02d}}/{{EPOCHS}} | "
        f"Train Loss: {{train_loss:.4f}} | Train Acc: {{train_acc:.4f}} | "
        f"Val Loss: {{val_loss:.4f}} | Val Acc: {{val_acc:.4f}}"
    )

# Restore best model
model.load_state_dict(best_state)


# ============================================================
# 19. TEST EVALUATION
# ============================================================
test_loss, test_acc, y_true, y_prob = run_epoch(model, test_loader, criterion, optimizer=None)
y_pred = (y_prob >= 0.5).astype(int)

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auroc = roc_auc_score(y_true, y_prob)
except:
    auroc = np.nan

try:
    ll = log_loss(y_true, y_prob, labels=[0, 1])
except:
    ll = np.nan

print("
================ TEST METRICS ================" )
print(f"Accuracy : {{test_acc:.4f}}")
print(f"Log Loss : {{ll:.4f}}")
print(f"Precision: {{precision:.4f}}")
print(f"Recall   : {{recall:.4f}}")
print(f"F1-Score : {{f1:.4f}}")
print(f"AUROC    : {{auroc:.4f}}")


# ============================================================
# 20. TRAINING CURVES
# ============================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disaster", "Disaster"]
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d", ax=ax)
plt.title("ConvLSTM Confusion Matrix")
plt.show()


# ============================================================
# 22. PRECISION-RECALL CURVE
# ============================================================
pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
ap = auc(pr_recall, pr_precision)

plt.figure(figsize=(6, 5))
plt.plot(pr_recall, pr_precision, label=f"AP = {{ap:.4f}}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================
results = pd.DataFrame({
    "y_true": y_true.astype(int),
    "y_prob": y_prob,
    "y_pred": y_pred
})

results.to_csv("/content/convlstm_pytorch_predictions.csv", index=False)
print("Saved: /content/convlstm_pytorch_predictions.csv")


# ============================================================
# 24. FINAL DIAGNOSTICS
# ============================================================
print("
Final diagnostics:")
print("T =", T)
print("Number of active cells =", len(active_cells))
print("Train class distribution =", np.unique(y_train.astype(int), return_counts=True))
print("Done.")