"""Task 3 - choose alpha, train logistic regression on Set R and Set Q, save theta and tables/figures.

Run from the repo root:  python src/train_logistic.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logistic import cost, fit_logistic, predict
from metrics import accuracy
from prepare import FEATURES, prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIGS = os.path.join(RES, "figures")
MODELS = os.path.join(RES, "models")

ALPHAS = [0.01, 0.1, 1, 10]
N_ITERS = 2000
Q_NAMES = ["theta0 (intercept)", "Ia", "Ib", "Ic", "Va", "Vb", "Vc",
           "Ia^2", "Ib^2", "Ic^2", "Va^2", "Vb^2", "Vc^2"]


def main():
    for p in (FIGS, MODELS):
        os.makedirs(p, exist_ok=True)
    d = prepare()
    ytr, yva = d["ytr"], d["yva"]

    # sanity check from the brief: J(theta = 0) = log 2
    j0 = cost(d["R_tr"], ytr, np.zeros(d["R_tr"].shape[1]))
    assert abs(j0 - np.log(2)) < 1e-9, j0
    print(f"J(0) = {j0:.4f}  (log 2 = {np.log(2):.4f})")

    chosen, final = {}, {}
    for name, A_tr in [("R", d["R_tr"]), ("Q", d["Q_tr"])]:
        histories, valid = {}, []
        for a in ALPHAS:
            _, h = fit_logistic(A_tr, ytr, a, N_ITERS)
            histories[a] = h
            if np.all(np.diff(h) < 0):            # J must fall at every iteration
                valid.append(a)
        chosen[name] = max(valid)                 # largest alpha that behaves
        final[name] = fit_logistic(A_tr, ytr, chosen[name], N_ITERS)
        print(f"Set {name}: valid alphas {valid} -> chosen alpha = {chosen[name]}")

        plt.figure(figsize=(10, 6))
        for a in ALPHAS:
            plt.plot(histories[a], label=f"alpha = {a}")
        plt.xlabel("Iteration"); plt.ylabel("J(theta)"); plt.legend(); plt.grid(True)
        plt.title(f"Set {name} - J(theta) vs iteration")
        plt.savefig(f"{FIGS}/fig2_cost_set{name}_four_alphas.png", dpi=110, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(final[name][1]); plt.xlabel("Iteration"); plt.ylabel("J(theta)"); plt.grid(True)
        plt.title(f"Final Set {name} cost (alpha = {chosen[name]})")
        plt.savefig(f"{FIGS}/extra_final_set{name}_cost.png", dpi=110, bbox_inches="tight")
        plt.close()

    theta_R, theta_Q = final["R"][0], final["Q"][0]
    table2 = pd.DataFrame({
        "Feature set": ["Set R", "Set Q"],
        "Chosen alpha": [chosen["R"], chosen["Q"]],
        "Final J(theta)": [f"{final['R'][1][-1]:.4f}", f"{final['Q'][1][-1]:.4f}"],
        "Validation accuracy": [f"{accuracy(yva, predict(d['R_va'], theta_R)):.4f}",
                                f"{accuracy(yva, predict(d['Q_va'], theta_Q)):.4f}"]})
    table5 = pd.DataFrame({"Weight": Q_NAMES, "Value": [f"{t:.4f}" for t in theta_Q]})
    table2.to_csv(f"{RES}/table2_logreg.csv", index=False)
    table5.to_csv(f"{RES}/table5_theta_setQ.csv", index=False)

    # saved model: theta + the TRAINING statistics needed to reproduce the standardisation
    with open(f"{MODELS}/logreg_q.json", "w") as f:
        json.dump({"theta": theta_Q.tolist(), "q_mean": d["qmu"].tolist(), "q_std": d["qsd"].tolist(),
                   "mean6": d["mu"].tolist(), "std6": d["sd"].tolist(),
                   "feature_names": FEATURES}, f, indent=2)
    with open(f"{MODELS}/logreg_r.json", "w") as f:
        json.dump({"theta": theta_R.tolist(), "mean6": d["mu"].tolist(), "std6": d["sd"].tolist(),
                   "feature_names": FEATURES}, f, indent=2)

    print(table2.to_string(index=False))
    print(table5.to_string(index=False))


if __name__ == "__main__":
    main()
