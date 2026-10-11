from pathlib import Path
import urllib.request
import zipfile
import io
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "detect dataset.csv"
ALT_DATA_PATH = ROOT / "data" / "detect_dataset.csv"
MODEL_PATH = ROOT / "models" / "best_rbf_svm.joblib"
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
FEATURE_NAMES = FEATURES

# Exact Set-Q parameters printed by the user's Colab notebook.
Q_THETA = np.array([
    24.920914122367883,
    2.3272915930675713, -5.358083867219878, -1.7365031309695322,
    -0.533568713039086, 0.4167264098060003, 0.09857962774795893,
    18.891238299142962, 4.754255911649336, 13.956918039601511,
    -6.027636434055077, -6.464258365360719, -6.140745456717821
], dtype=float)

Q_MEAN = np.array([
    5.045347551502079, -28.355203596964987, 24.988539440415263,
    0.012428791667499989, -0.01667949190930563, 0.004250643307361116,
    139329.54040862274, 129444.57635822908, 92795.45598397453,
    0.11894592583035495, 0.1289679605880668, 0.12185293267573157
], dtype=float)
Q_STD = np.array([
    373.234624434426, 358.6649673263394, 303.59681961543765,
    0.3446613569404622, 0.3587335433684926, 0.3490485134006529,
    240450.61182597393, 233126.91310538648, 203220.0631568904,
    0.1277376195907853, 0.1332313210772683, 0.12418851054151303
], dtype=float)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def predict_set_q(raw_values):
    raw_values = np.asarray(raw_values, dtype=float).reshape(6)
    q_raw = np.concatenate([raw_values, raw_values ** 2])
    q_scaled = (q_raw - Q_MEAN) / np.where(Q_STD == 0, 1, Q_STD)
    x = np.concatenate([[1.0], q_scaled])
    probability = float(sigmoid(x @ Q_THETA))
    return int(probability >= 0.5), probability

def load_or_train_rbf():
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and {"model", "mean", "std"} <= set(bundle):
            return bundle
        # Handle a model saved by the notebook, but it needs preprocessing values.
        # Do not silently guess these values.
        raise ValueError("Found a model file without its training mean/std. Remove it and retrain from the CSV.")
    data_path = DATA_PATH if DATA_PATH.exists() else ALT_DATA_PATH
    if not data_path.exists():
        # Best-effort attempt to fetch the public Kaggle dataset. Kaggle may require
        # sign-in/API credentials in some environments, so manual placement remains supported.
        url = "https://www.kaggle.com/api/v1/datasets/download/esathyaprakash/electrical-fault-detection-and-classification"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_members = [name for name in archive.namelist()
                               if Path(name).name.lower() in {"detect dataset.csv", "detect_dataset.csv"}]
                if not csv_members:
                    raise ValueError("Kaggle archive did not contain detect dataset.csv")
                preferred = next((name for name in csv_members
                                  if Path(name).name.lower() == "detect dataset.csv"), csv_members[0])
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                DATA_PATH.write_bytes(archive.read(preferred))
                data_path = DATA_PATH
        except Exception as download_error:
            raise FileNotFoundError(
                "Could not automatically download the Kaggle CSV (Kaggle may require login). "
                f"Please download detect dataset.csv from the assignment Kaggle page and place it in {DATA_PATH.parent}. "
                f"Download detail: {download_error}"
            ) from download_error
    from sklearn.svm import SVC

    df = pd.read_csv(data_path)
    df = df.dropna(axis=1, how="all")
    if "Output (S)" in df.columns:
        df = df.rename(columns={"Output (S)": "fault"})
    if "fault" not in df.columns:
        raise ValueError("CSV must contain label column 'Output (S)' or 'fault'.")
    for col in FEATURES + ["fault"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + ["fault"])
    df = df[df["fault"].isin([0, 1])].copy()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["fault"].to_numpy(dtype=int)

    # Match the assignment's seeded 60/20/20 split and training-only standardization.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
    train_idx = idx[:n_train]
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)
    std = np.where(std == 0, 1, std)
    X_train = (X[train_idx] - mean) / std

    model = SVC(kernel="rbf", C=100, gamma=1)
    model.fit(X_train, y[train_idx])
    bundle = {"model": model, "mean": mean, "std": std, "C": 100, "gamma": 1}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle
