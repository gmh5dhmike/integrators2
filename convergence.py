#!/usr/bin/env python3
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from ndsphere import estimate_volume, true_volume

def run_sweep(ds=(3,5,10), pmin=6, pmax=18, r=1.0, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    for d in ds:
        for p in range(pmin, pmax + 1):
            N = 2**p
            vol, sigma, rel_err, inside = estimate_volume(d, N, r, rng)
            rows.append({
                "d": d,
                "N": N,
                "sqrtN": math.sqrt(N),
                "estimate": vol,
                "true": true_volume(d, r),
                "fractional_error": rel_err,
                "sigma": sigma,
                "sigma_frac": sigma / true_volume(d, r),
                "inside": inside,
                "r": r,
            })
    return rows

def save_csv(rows, path="convergence.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def make_plot(rows, path="convergence.png"):
    plt.figure()
    for d in sorted(set(r["d"] for r in rows)):
        sub = [r for r in rows if r["d"] == d]
        x = [r["sqrtN"] for r in sub]
        y = [r["fractional_error"] for r in sub]
        yerr = [r["sigma_frac"] for r in sub]
        plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label=f"d={d}")
    plt.xlabel(r"$\sqrt{N}$")
    plt.ylabel("fractional error")
    plt.title("MC convergence for unit d-sphere (hit-or-miss)")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print(f"Wrote {path}")

if __name__ == "__main__":
    rows = run_sweep(ds=(3,5,10), pmin=6, pmax=18, r=1.0, seed=42)  # set pmax=24 for final
    save_csv(rows, "convergence.csv")
    make_plot(rows, "convergence.png")
    print("Wrote convergence.csv")
