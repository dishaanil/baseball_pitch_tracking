# Baseball Pitch Tracking (Multimodal ML Pipeline)

This project explores baseball pitch tracking and prediction using a combination of:

- **Tabular + 2D modeling (baseline)**
- **3D spatiotemporal video modeling (advanced)**
- **Model ensembling for improved performance**

The goal is to leverage both structured pitch features and video-based motion patterns to predict pitch characteristics.

---

## Key Highlights

- Baseline pipeline using **ResNet18 + tabular features**
- Advanced pipeline using **3D CNN (r3d_18 with pretrained weights)**
- Multimodal fusion (video + tabular data)
- Ensemble modeling for improved accuracy
- End-to-end training → validation → inference → submission

---

## Project Structure

```text
baseball_project/
├── src/                         # Baseline pipeline (2D / tabular)
│   ├── config.py
│   ├── dataset.py
│   ├── models.py               # uses ResNet18
│   ├── train_deep.py
│   ├── train_tabular.py
│   ├── infer_ensemble.py
│   └── utils_video.py
│
├── src_3d/                     # Advanced 3D pipeline
│   ├── config_3d.py
│   ├── dataset_3d.py
│   ├── models_3d.py           # uses r3d_18 + R3D_18_Weights
│   ├── train_deep_3d.py
│   ├── train_tabular_3d.py
│   ├── tune_ensemble_3d.py
│   ├── infer_ensemble_3d.py
│   └── utils_video_3d.py
│
├── baseball_kaggle_dataset_trimmed_only/   # (NOT INCLUDED IN REPO)
├── test_submission_template.csv
├── README.md
└── .gitignore
```

---

## Modeling Approach

### 1. Baseline Pipeline (`src/`)
- Uses **ResNet18** for feature extraction
- Combines video features with tabular baseball data
- Includes:
  - Deep learning model
  - Tabular model (XGBoost)
  - Ensemble inference

---

### 2. 3D Pipeline (`src_3d/`)
- Uses **r3d_18 (3D ResNet)** with:
  - `R3D_18_Weights` (pretrained on Kinetics-400)
- Learns **spatiotemporal motion directly from pitch videos**
- Combines:
  - Video features (3D CNN)
  - Tabular features
- Multi-task prediction:
  - Pitch classification (strike / ball)
  - Zone prediction
  - Plate location (x, z)

---

### 3. Ensemble Strategy
- Combines:
  - Deep learning predictions
  - Tabular model predictions
- Weighted averaging improves final performance

---

## Dataset

This project uses the **Baseball Pitch Tracking dataset** from Kaggle.

🔗 Dataset link:  
https://www.kaggle.com/competitions/baseball-pitch-tracking-cs-gy-6643/data

---

## How to Download Dataset

### Option 1: Manual Download
1. Go to the Kaggle link above
2. Download the dataset ZIP
3. Extract it

---

### Option 2: Kaggle CLI

```bash
pip install kaggle
```

Set up API key (`~/.kaggle/kaggle.json`), then:

```bash
kaggle competitions download -c baseball-pitch-tracking-cs-gy-6643
unzip baseball-pitch-tracking-cs-gy-6643.zip
```

---

## Expected Folder Structure

Place the dataset in the project root like this:

```text
baseball_project/
├── baseball_kaggle_dataset_trimmed_only/
│   ├── data/
│   │   ├── train_ground_truth.csv
│   │   ├── test_features.csv
│   │
│   ├── train_trimmed/      # training videos
│   ├── test/               # test videos
```

Important:
- The dataset folder **must be named exactly**:
  
  ```
  baseball_kaggle_dataset_trimmed_only
  ```

- This is required because paths are hardcoded in config files.

---

## Setup

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Baseline pipeline
```bash
python src/train_tabular.py
python src/train_deep.py
python src/infer_ensemble.py
```

---

### 3D pipeline
```bash
python src_3d/train_tabular_3d.py
python src_3d/train_deep_3d.py
python src_3d/tune_ensemble_3d.py
python src_3d/infer_ensemble_3d.py
```

---

## Notes

- Dataset is **not included** due to size constraints
- Outputs are saved in:
  - `outputs_3d/` (for 3D pipeline)
- GPU is recommended for 3D training
- Paths and hyperparameters are configurable in `config.py` and `config_3d.py`

---

## Future Improvements

- Better model ensembling strategies
- Experiment tracking (Weights & Biases / MLflow)
- Hyperparameter tuning
- Real-time inference pipeline
- Visualization of pitch trajectories

---

## Summary

This project demonstrates:

- Multimodal ML (tabular + video)
- Deep learning + classical ML integration
- 3D CNNs for motion understanding
- End-to-end ML pipeline design

---
