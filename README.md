# 🌍 ConvLSTM Spatio-Temporal Disaster Prediction

A deep learning project that uses **Convolutional LSTM (ConvLSTM)** networks to predict natural disasters across space and time, with an interactive **Streamlit dashboard** for visualization and analysis.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds a **ConvLSTM-based binary classifier** that predicts whether a disaster will occur in a specific geographic grid cell within a future time horizon (e.g., 6 months ahead), using 12 months of historical spatio-temporal data.

### Key Features

- 🧠 **3-Layer ConvLSTM** architecture for spatio-temporal feature extraction
- 🗺️ **Spatial gridding** (1° × 1°) with 3×3 local patches
- 📊 **Interactive Streamlit Dashboard** with 5 tabs:
  - Dashboard (metrics, confusion matrix, distributions)
  - Training Curves (accuracy & loss plots)
  - Geo Map (world heatmap of disaster risk)
  - Analysis (ROC, PR curves, threshold sensitivity)
  - Downloads (predictions, metrics, history as CSV)
- ⚖️ **Weighted sampling** to handle class imbalance
- 🔧 **Tunable hyperparameters** via sidebar controls

---

## 🗂️ Project Structure

```
convlstm_disaster_pred/
├── convlstm_disaster_pred.py   # Original training script (Colab/CLI)
├── app.py                       # Streamlit frontend dashboard
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/devbharti466/convlstm_disaster_pred.git
cd convlstm_disaster_pred
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

### 4. Upload Your Dataset

Upload your `.xlsx` file via the sidebar. The app expects these columns:

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

---

## ⚙️ Configurable Hyperparameters

All tunable from the Streamlit sidebar:

| Parameter | Default | Range |
|-----------|---------|-------|
| Grid Size | 1.0° | 0.5–5.0° |
| Patch Size | 3×3 | 3, 5, 7 |
| Sequence Length | 12 months | 3–24 |
| Prediction Horizon | 6 months | 1–12 |
| Epochs | 8 | 2–30 |
| Batch Size | 32 | 16–128 |
| Learning Rate | 1e-3 | 1e-4 to 1e-2 |
| Dropout | 0.3 | 0.0–0.7 |
| Threshold | 0.5 | 0.1–0.9 |

---

## 🏗️ Model Architecture

```
ConvLSTMClassifier(
  (convlstm): ConvLSTM(
    3 layers × ConvLSTMCell(hidden=64, kernel=3×3)
  )
  (dropout): Dropout(p=0.3)
  (fc): Linear(64×3×3 → 1)
)
```

---

## 📈 Metrics Tracked

- Accuracy
- Precision
- Recall
- F1-Score
- AUROC
- Log Loss

---

## 🌐 Deployment Options

### Streamlit Community Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy

### Hugging Face Spaces
1. Create a new Space with Streamlit SDK
2. Push your files
3. App auto-deploys

### Docker
```bash
docker build -t disaster-pred .
docker run -p 8501:8501 disaster-pred
```

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

Built with ❤️ using PyTorch & Streamlit
