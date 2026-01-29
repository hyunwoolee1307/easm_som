"""Minimal SOM runner. Expects preprocessed data saved as NumPy .npy arrays."""
import argparse
import numpy as np
from som import Som


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SOM on preprocessed u850 anomalies")
    parser.add_argument("--x", required=True, help="Path to .npy array with shape (nd, nt)")
    parser.add_argument("--m1", type=int, default=3)
    parser.add_argument("--m2", type=int, default=3)
    parser.add_argument("--nter", type=int, default=500)
    args = parser.parse_args()

    x = np.load(args.x)
    if x.ndim != 2:
        raise ValueError("Expected x with shape (nd, nt)")
    nd, nt = x.shape

    w = Som(
        x=x,
        nd=nd,
        nt=nt,
        m1=args.m1,
        m2=args.m2,
        nter=args.nter,
        alpha0=0.5,
        taua=200.0,
        alphamin=0.01,
        sigma0=1.5,
        taus=200.0,
        normalize_weights=True,
        seed=1234,
    )
    np.save("results/som_weights.npy", w)


if __name__ == "__main__":
    main()
