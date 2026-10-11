"""Task 2 - exploratory plots (raw, unstandardised data)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare import prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "results", "figures")


def make_plots(df, figs=FIGS):
    os.makedirs(figs, exist_ok=True)
    first = df.iloc[:2000].reset_index(drop=True)

    plt.figure(figsize=(15, 6))
    for c in ["Ia", "Ib", "Ic"]:
        plt.plot(first[c], label=c)
    for i in np.where(first["fault"].to_numpy() == 1)[0]:
        plt.axvspan(i, i + 1, alpha=0.25, color="tab:blue", linewidth=0)
    plt.xlabel("Row number"); plt.ylabel("Current"); plt.legend(); plt.grid(True)
    plt.title("Ia, Ib, Ic - first 2000 rows (shaded: fault = 1)")
    plt.savefig(f"{figs}/fig1_currents_fault_shaded.png", dpi=110, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(9, 7))
    for lab, name in [(0, "No fault (0)"), (1, "Fault (1)")]:
        s = df[df.fault == lab]
        plt.scatter(s["Ia"], s["Ib"], alpha=0.5, label=name)
    plt.xlabel("Ia"); plt.ylabel("Ib"); plt.legend(); plt.grid(True)
    plt.title("Ia against Ib")
    plt.savefig(f"{figs}/fig1_scatter_Ia_Ib.png", dpi=110, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(10, 6))
    for lab, name in [(0, "No fault (0)"), (1, "Fault (1)")]:
        plt.hist(df.loc[df.fault == lab, "Ia"], bins=50, alpha=0.6, label=name)
    plt.xlabel("Ia"); plt.ylabel("Frequency"); plt.legend(); plt.grid(True)
    plt.title("Histogram of Ia")
    plt.savefig(f"{figs}/fig1_hist_Ia.png", dpi=110, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    make_plots(prepare()["df"])
    print("EDA figures written to", FIGS)
