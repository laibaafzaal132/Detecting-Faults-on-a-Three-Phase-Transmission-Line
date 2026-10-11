"""Task 1 - cleaning, exact 60/20/20 split (seed 42) and standardisation (training statistics only)."""
import numpy as np
import pandas as pd

from load_data import load_data

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def clean(raw):
    """Drop completely empty columns, rename the label to 'fault', force numeric measurements."""
    empty_cols = raw.columns[raw.isnull().all()].tolist()
    df = raw.drop(columns=empty_cols).rename(columns={"Output (S)": "fault"})
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    assert df.isnull().sum().sum() == 0, "missing values remain after cleaning"
    return df, empty_cols


def split(df):
    """The exact split from the brief."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
    tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    return df.iloc[tr].copy(), df.iloc[va].copy(), df.iloc[te].copy()


def add_intercept(X):
    return np.column_stack((np.ones(len(X)), X))


def prepare(path=None):
    """Run all of Task 1 and return everything later tasks need in one dict."""
    raw = load_data(path) if path else load_data()
    df, empty_cols = clean(raw)
    train_df, val_df, test_df = split(df)

    Xtr, ytr = train_df[FEATURES].to_numpy(), train_df["fault"].to_numpy()
    Xva, yva = val_df[FEATURES].to_numpy(), val_df["fault"].to_numpy()
    Xte, yte = test_df[FEATURES].to_numpy(), test_df["fault"].to_numpy()

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)           # training statistics only
    Str, Sva, Ste = [(X - mu) / sd for X in (Xtr, Xva, Xte)]

    # Set Q: squares are taken from the RAW values, then all 12 columns are standardised
    Qraw = [np.column_stack((X, X ** 2)) for X in (Xtr, Xva, Xte)]
    qmu, qsd = Qraw[0].mean(axis=0), Qraw[0].std(axis=0)
    Qtr, Qva, Qte = [(X - qmu) / qsd for X in Qraw]

    return dict(raw_shape=raw.shape, empty_cols=empty_cols, df=df,
                train_df=train_df, val_df=val_df, test_df=test_df,
                ytr=ytr, yva=yva, yte=yte,
                Xtr=Xtr, Xva=Xva, Xte=Xte, mu=mu, sd=sd,
                Str=Str, Sva=Sva, Ste=Ste,
                R_tr=add_intercept(Str), R_va=add_intercept(Sva), R_te=add_intercept(Ste),
                qmu=qmu, qsd=qsd,
                Q_tr=add_intercept(Qtr), Q_va=add_intercept(Qva), Q_te=add_intercept(Qte))


def data_prep_table(d):
    """Table 1."""
    df, ytr, yva, yte = d["df"], d["ytr"], d["yva"], d["yte"]
    return pd.DataFrame({
        "Item": ["Raw rows / raw columns", "Columns removed (names)",
                 "Missing values after cleaning", "Fault rows / no-fault rows",
                 "Train / validation / test rows",
                 "Fault fraction in train / validation / test"],
        "Value": [f"{d['raw_shape'][0]} / {d['raw_shape'][1]}", ", ".join(d["empty_cols"]),
                  int(df.isnull().sum().sum()),
                  f"{int((df.fault == 1).sum())} / {int((df.fault == 0).sum())}",
                  f"{len(d['train_df'])} / {len(d['val_df'])} / {len(d['test_df'])}",
                  f"{ytr.mean():.4f} / {yva.mean():.4f} / {yte.mean():.4f}"]})


if __name__ == "__main__":
    d = prepare()
    print(data_prep_table(d).to_string(index=False))
    print("std. train mean ~", np.round(d["Str"].mean(axis=0), 6), " std ~", np.round(d["Str"].std(axis=0), 6))
