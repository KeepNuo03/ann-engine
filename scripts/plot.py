"""Recall@10 vs QPS (log scale), one line per (algo, M, efc, threads), ann-benchmarks style."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="results/recall_qps.png")
    ap.add_argument("--threads", type=int, default=1, help="which thread count to plot")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    df = df[(df.ef > 0) & (df.threads == args.threads)]
    # If a config was run more than once, keep the latest row.
    df = df.drop_duplicates(subset=["algo", "M", "efc", "ef", "threads"], keep="last")

    fig, ax = plt.subplots(figsize=(7, 5))
    for (algo, M, efc), g in df.groupby(["algo", "M", "efc"]):
        g = g.sort_values("recall10")
        name = {"hnsw_v3": "ours, v3", "hnsw_v0": "ours, v0", "hnsw": "ours", "hnswlib": "hnswlib"}.get(algo, algo)
        ax.plot(g.recall10, g.qps, marker="o", label=f"{name} (M={M}, efc={efc})")
    ax.set_yscale("log")
    ax.set_xlim(left=max(0.0, min(0.8, df.recall10.min() - 0.02)), right=1.0)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (log)")
    cpu = df.cpu.iloc[0] if len(df) else ""
    ax.set_title(f"SIFT1M, {args.threads} thread(s)\n{cpu}", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
