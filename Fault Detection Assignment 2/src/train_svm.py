"""Task 4 - linear and RBF SVMs on Set R; choose by validation accuracy; save the best of each kernel.

Run from the repo root:  python src/train_svm.py
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

from metrics import accuracy
from prepare import prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figures")
MODELS = os.path.join(RES, "models")

C_VALUES = [0.1, 1, 10, 100]
GAMMAS = [0.1, 1]


def main():
    for p in (FIGS, MODELS):
        os.makedirs(p, exist_ok=True)
    d = prepare()
    Str, Sva, ytr, yva = d["Str"], d["Sva"], d["ytr"], d["yva"]

    rows, lin_models, rbf_models = [], {}, {}
    for C in C_VALUES:
        m = SVC(kernel="linear", C=C).fit(Str, ytr)
        lin_models[C] = m
        rows.append(["linear", C, "-", accuracy(yva, m.predict(Sva)), int(m.n_support_.sum())])
    for C in C_VALUES:
        for g in GAMMAS:
            m = SVC(kernel="rbf", C=C, gamma=g).fit(Str, ytr)
            rbf_models[(C, g)] = m
            rows.append(["rbf", C, g, accuracy(yva, m.predict(Sva)), int(m.n_support_.sum())])

    svm_df = pd.DataFrame(rows, columns=["Kernel", "C", "gamma", "Validation accuracy", "Support vectors"])
    # best = highest validation accuracy; ties are broken towards the larger C (deterministic)
    rank = ["Validation accuracy", "C"]
    best_lin = svm_df[svm_df.Kernel == "linear"].sort_values(rank, ascending=False).iloc[0]
    best_rbf = svm_df[svm_df.Kernel == "rbf"].sort_values(rank, ascending=False).iloc[0]
    joblib.dump(lin_models[best_lin["C"]], f"{MODELS}/best_linear_svm.joblib")
    joblib.dump(rbf_models[(best_rbf["C"], best_rbf["gamma"])], f"{MODELS}/best_rbf_svm.joblib")

    table3 = svm_df.copy()
    table3["Validation accuracy"] = table3["Validation accuracy"].map("{:.4f}".format)
    table3.to_csv(f"{RES}/table3_svm_validation.csv", index=False)

    plt.figure(figsize=(9, 6))
    lin = svm_df[svm_df.Kernel == "linear"]
    plt.plot(lin.C, lin["Validation accuracy"], "o-", label="linear")
    for g in GAMMAS:
        s = svm_df[(svm_df.Kernel == "rbf") & (svm_df.gamma == g)]
        plt.plot(s.C, s["Validation accuracy"], "o-", label=f"RBF, gamma = {g}")
    plt.xscale("log"); plt.xlabel("C (log scale)"); plt.ylabel("Validation accuracy")
    plt.title("SVM validation accuracy against C"); plt.legend(); plt.grid(True, which="both")
    plt.savefig(f"{FIGS}/fig3_svm_val_accuracy_vs_C.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(table3.to_string(index=False))
    print(f"Best linear: C={best_lin['C']:g} | Best RBF: C={best_rbf['C']:g}, gamma={best_rbf['gamma']:g}")


if __name__ == "__main__":
    main()
