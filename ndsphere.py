#!/usr/bin/env python3
import sys
import math
import numpy as np

def true_volume(d: int, r: float) -> float:
    """Analytic volume of a d-ball of radius r."""
    return (math.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1.0) * (r ** d)

def estimate_volume(d: int, N: int, r: float, rng: np.random.Generator | None = None):
    """
    Hit-or-miss Monte Carlo for d-dimensional ball volume.
    Returns (volume_est, stat_uncertainty, rel_error, inside)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample uniformly in the hypercube [-r, r]^d
    X = rng.uniform(-r, r, size=(N, d))
    sq = np.sum(X * X, axis=1)
    inside = int(np.count_nonzero(sq <= r * r))

    cube_vol = (2.0 * r) ** d
    p_hat = inside / N
    vol_hat = cube_vol * p_hat

    # Bernoulli standard error on p, scaled by cube volume
    sigma = cube_vol * math.sqrt(max(p_hat * (1.0 - p_hat) / N, 0.0))

    v_true = true_volume(d, r)
    rel_err = abs(vol_hat - v_true) / v_true

    return vol_hat, sigma, rel_err, inside

def main():
    # Keep your original CLI/format exactly
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <int1> <int2> <double>")
        sys.exit(1)

    try:
        d = int(sys.argv[1])   # dimension
        N = int(sys.argv[2])   # number of samples
        r = float(sys.argv[3]) # radius
    except ValueError:
        print("Error: Please provide valid integers and doubles as arguments.")
        sys.exit(1)

    volume, stdev, relerror, _ = estimate_volume(d, N, r)

    # Do not change the format below
    print(f"(r): {r}")
    print(f"(d,N): {d} {N}")
    print(f"volume: {volume}")
    print(f"stat uncertainty: {stdev}")
    print(f"relative error: {relerror}")

if __name__ == "__main__":
    main()
