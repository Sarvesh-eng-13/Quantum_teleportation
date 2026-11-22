"""
experiment.py - Monte Carlo analysis + graphs.

Graph 1: fidelity vs entanglement quality p   (simulation vs theory (1+p)/2)
Graph 2: success rate vs p for different channel conditions (loss + classical bit-flip)
Both graphs show the classical limit F = 2/3.
"""
import numpy as np
import matplotlib.pyplot as plt
from teleport import run_trials, CLASSICAL_LIMIT

TRIALS = 1000                                  # Monte Carlo trials per point
P_VALUES = np.linspace(0, 1, 21)               # entanglement quality 0, 0.05, ... 1


def main(show=True):
    # ---------------- Graph 1: fidelity vs p (no loss, no bit-flip) ----------
    sim_f = [run_trials(p, 0.0, 0.0, TRIALS, seed=1)[0] for p in P_VALUES]

    fig1, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(P_VALUES, sim_f, "o", label="Simulation (1000 trials)")
    ax.plot(P_VALUES, (1 + P_VALUES) / 2, "-", label="Theory F = (1 + p) / 2")
    ax.axhline(CLASSICAL_LIMIT, color="red", linestyle="--", label="Classical limit F = 2/3")
    ax.axvline(1 / 3, color="gray", linestyle=":", label="Threshold p = 1/3")
    ax.set_xlabel("Entanglement quality p  (1 = perfect, 0 = pure noise)")
    ax.set_ylabel("Average fidelity F")
    ax.set_title("Teleportation fidelity vs noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("graph1_fidelity_vs_noise.png", dpi=150)

    # ---------------- Graph 2: success rate vs p for channel conditions ------
    conditions = [
        (0.0, 0.0, "No loss, no bit-flip"),
        (0.1, 0.05, "10% loss, 5% bit-flip"),
        (0.3, 0.10, "30% loss, 10% bit-flip"),
    ]
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for loss, flip, label in conditions:
        rates = [run_trials(p, loss, flip, TRIALS, seed=2)[1] * 100 for p in P_VALUES]
        ax2.plot(P_VALUES, rates, "o-", label=label)
    ax2.set_xlabel("Entanglement quality p  (1 = perfect, 0 = pure noise)")
    ax2.set_ylabel("Success rate (%)   [arrived AND F > 2/3]")
    ax2.set_title("Transmission success rate vs noise")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("graph2_success_rate.png", dpi=150)

    print("Saved graph1_fidelity_vs_noise.png and graph2_success_rate.png")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
