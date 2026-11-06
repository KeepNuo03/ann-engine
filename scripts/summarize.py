"""Builds results/summary.md: ablation table and recall ~0.95 / ~0.99 summary from the CSVs."""
import os

import pandas as pd

R = "results"


def load(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fmt_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def ablation(out):
    df = load("ablation.csv")
    if df is None:
        return
    labels = {
        "hnsw_v0": "v0: unordered_set visited",
        "hnsw_v1": "v1: epoch visited list",
        "hnsw_v2": "v2: + interleaved level-0 layout",
        "hnsw_v3": "v3: + software prefetch",
        "hnswlib": "hnswlib (reference)",
    }
    for ef in sorted(df.ef.unique()):
        g = df[df.ef == ef].drop_duplicates("algo", keep="last").set_index("algo")
        rows, prev = [], None
        for a in ["hnsw_v0", "hnsw_v1", "hnsw_v2", "hnsw_v3", "hnswlib"]:
            if a not in g.index:
                continue
            r = g.loc[a]
            rel = "" if prev is None or a == "hnswlib" else f"{(r.qps / prev - 1) * 100:+.1f}%"
            vs_lib = f"{r.qps / g.loc['hnswlib'].qps * 100:.0f}%" if "hnswlib" in g.index else ""
            rows.append({"v": labels[a], "qps": f"{r.qps:.0f}", "rec": f"{r.recall10:.4f}",
                         "p50": f"{r.p50_us:.0f}", "p99": f"{r.p99_us:.0f}", "rel": rel, "lib": vs_lib})
            if a != "hnswlib":
                prev = r.qps
        t = pd.DataFrame(rows)
        out.append(f"### Ablation at ef={ef} (single thread)\n")
        out.append(fmt_table(t, ["v", "qps", "rec", "p50", "p99", "rel", "lib"],
                             ["Version", "QPS", "Recall@10", "p50 (us)", "p99 (us)",
                              "vs previous", "vs hnswlib"]))
        out.append("")


def summary(out):
    frames = [f for f in (load("hnsw.csv"), load("hnswlib.csv"), load("ablation.csv")) if f is not None]
    if not frames:
        return
    alldf = pd.concat(frames, ignore_index=True)
    df = alldf[(alldf.threads == 1) & (alldf.ef > 0)]
    # Our current implementation is v3; older unlabeled "hnsw" rows predate the variants.
    show = {"hnsw_v3": "ours (v3)", "hnsw_v0": "ours (v0, naive visited set)", "hnswlib": "hnswlib"}

    def build_time(is_lib):
        sel = alldf[(alldf.build_s > 0) & ((alldf.algo == "hnswlib") if is_lib else
                                          (alldf.algo.str.startswith("hnsw") & (alldf.algo != "hnswlib")))]
        return f"{sel.build_s.iloc[-1]:.1f}" if len(sel) else ""

    rows = []
    for target in (0.95, 0.99):
        for algo, label in show.items():
            g = df[df.algo == algo]
            ok = g[g.recall10 >= target].sort_values("ef")
            if ok.empty:
                continue
            r = ok.drop_duplicates("ef", keep="last").iloc[0]
            rows.append({"t": f">= {target}", "algo": label, "ef": int(r.ef), "rec": f"{r.recall10:.4f}",
                         "qps": f"{r.qps:.0f}", "p99": f"{r.p99_us:.0f}",
                         "build": build_time(algo == "hnswlib")})
    out.append("### Summary (single thread, smallest ef reaching the recall target)\n")
    out.append(fmt_table(pd.DataFrame(rows), ["t", "algo", "ef", "rec", "qps", "p99", "build"],
                         ["Recall target", "Implementation", "ef", "Recall@10", "QPS", "p99 (us)",
                          "Index build (s, all cores)"]))
    out.append("")


def main():
    out = ["# Results\n"]
    for f in ("hnsw.csv", "hnswlib.csv", "ablation.csv"):
        d = load(f)
        if d is not None and len(d):
            out.append(f"Machine: {d.cpu.iloc[0]}, {d.compiler.iloc[0]}, SIMD {d.simd.iloc[0]}\n")
            break
    summary(out)
    ablation(out)
    text = "\n".join(out)
    with open(os.path.join(R, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
