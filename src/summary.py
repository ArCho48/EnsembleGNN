#!/usr/bin/env python3
import os
import math
import pdb
import json
import argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def _read_mses_for_method(method_dir: str):
    """
    Returns:
      - dict: {subrun_key -> mse} where subrun_key is the subfolder name (e.g., "rep0fold3").
              If duplicate subfolder names exist, they are disambiguated as "name#2", "name#3", etc.
    """
    mse_by_key = {}
    name_counts = Counter()

    if not os.path.isdir(method_dir):
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    for sub in sorted(os.listdir(method_dir)):
        sub_path = os.path.join(method_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        metrics_path = os.path.join(sub_path, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue

        # Handle possible duplicate subfolder names by suffixing counters
        name_counts[sub] += 1
        key = sub if name_counts[sub] == 1 else f"{sub}#{name_counts[sub]}"

        with open(metrics_path, "r") as f:
            m = json.load(f)
        rmse = m.get("rmse")
        if rmse is None:
            mse = m.get("mse")
            mse_by_key[key] = mse
            rmse = math.sqrt(float(mse))
            # raise KeyError(f"'rmse' not found in {metrics_path}")
        else:
            mse_by_key[key] = float(rmse) ** 2

    if not mse_by_key:
        raise FileNotFoundError(f"No metrics.json files found under {method_dir}")

    return mse_by_key

def summarize_methods(root_dir: str, methods=None):
    """
    Walks the folder structure under `root_dir`:
      root_dir/
        GCN|MLP|XGB|RF|SVM/
          repXfoldY.../metrics.json

    Returns:
      summary_df (2 x M): mean/std rows, methods as columns
      mse_dict (dict): {method -> {subrun_key -> mse}}
    """
    if methods is None:
        methods = ["SAGE", "XGB"]#, "GCN", "MLP", "RF", "SVM"]

    mse_dict = {}
    for method in methods:
        method_dir = os.path.join(root_dir, method)
        mse_dict[method] = _read_mses_for_method(method_dir)

    # Create summary table (mean/std) for each method using all its available subruns
    summary = {}
    for method, kv in mse_dict.items():
        vals = list(kv.values())
        summary[method] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),  # population std to match your snippet's np.std
        }
    pdb.set_trace()
    summary_df = pd.DataFrame(summary).loc[["mean", "std"]]
    return summary_df, mse_dict

from scipy.stats import wilcoxon
import numpy as np
import pandas as pd

def _holm_bonferroni(pairs, m):
    """
    Holm–Bonferroni adjustment. `pairs` is list of (key, pval).
    Returns dict key -> adjusted p.
    """
    # Filter NaNs but keep mapping
    valid = [(k, p) for (k, p) in pairs if p == p]  # p==p filters NaN
    # Sort by ascending p
    valid.sort(key=lambda x: x[1])
    adj = {}
    for i, (k, p) in enumerate(valid, start=1):
        adj_p = min((m - i + 1) * p, 1.0)
        adj[k] = adj_p
    # Put NaNs back unchanged
    for (k, p) in pairs:
        if k not in adj:
            adj[k] = np.nan
    return adj

def pairwise_wilcoxon(mse_dict: dict, adjust: str | None = "holm"):
    """
    Paired Wilcoxon tests between methods using only intersecting subrun keys.
    Adds direction, median diff, and win rates.
    
    Parameters
    ----------
    mse_dict : dict
        {method -> {subrun_key -> mse}}
    adjust : {"holm", None}
        Optional multiple-comparison correction for p-values.

    Returns
    -------
    tests : dict
        {
          "A_vs_B": {
            "statistic": float,
            "pvalue_raw": float,
            "pvalue_adj": float or NaN,
            "n_pairs": int,
            "median_diff": float,   # median of (A - B); negative => A better
            "win_rate_A": float,    # fraction A<B across pairs
            "win_rate_B": float,    # fraction B<A across pairs
            "direction": "A<B" | "A>B" | "tie/nd"  # nd = non-different
          },
          ...
        }
    pval_matrix : pd.DataFrame
        Symmetric matrix of (adjusted if adjust given else raw) p-values.
    direction_matrix : pd.DataFrame
        Matrix with arrows: "↑" (row better), "↓" (col better), "→" (no diff / tie).
    annotated_matrix : pd.DataFrame
        Strings like "↓ p=0.000061".
    """
    methods = list(mse_dict.keys())
    tests = {}
    pvals_raw_pairs = []  # (pair_key, p)
    pair_keys = []

    # First pass: compute raw stats per pair
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a_name, b_name = methods[i], methods[j]
            a_map, b_map = mse_dict[a_name], mse_dict[b_name]
            common = sorted(set(a_map).intersection(b_map))

            if len(common) == 0:
                stat = p_raw = np.nan
                med_diff = np.nan
                wr_a = wr_b = np.nan
                direction = "tie/nd"
            else:
                a = np.array([a_map[k] for k in common], dtype=float)
                b = np.array([b_map[k] for k in common], dtype=float)
                diffs = a - b
                # Win rates
                wr_a = float(np.mean(a < b)) if len(a) else np.nan
                wr_b = float(np.mean(b < a)) if len(a) else np.nan
                # Direction by median diff
                med_diff = float(np.median(diffs)) if len(diffs) else np.nan

                # Valid if any non-zero difference
                if np.any(np.abs(diffs) > 0):
                    try:
                        stat, p_raw = wilcoxon(a, b)
                    except ValueError:
                        stat = p_raw = np.nan
                else:
                    stat = p_raw = np.nan

                if p_raw == p_raw and p_raw < 0.05:  # significant
                    direction = "A<B" if med_diff < 0 else "A>B"
                else:
                    # Either not significant or NaN
                    direction = "tie/nd"

            key = f"{a_name}_vs_{b_name}"
            pair_keys.append((a_name, b_name, key))
            tests[key] = {
                "statistic": float(stat) if stat == stat else np.nan,
                "pvalue_raw": float(p_raw) if p_raw == p_raw else np.nan,
                "pvalue_adj": np.nan,  # fill later if adjust
                "n_pairs": int(len(common)),
                "median_diff": med_diff,  # negative => A better
                "win_rate_A": wr_a,
                "win_rate_B": wr_b,
                "direction": direction,
            }
            pvals_raw_pairs.append((key, tests[key]["pvalue_raw"]))

    # Multiple-comparison adjustment (Holm)
    if adjust == "holm":
        m = sum(1 for _, p in pvals_raw_pairs if p == p)
        adj_map = _holm_bonferroni(pvals_raw_pairs, m)
        for key, p_adj in adj_map.items():
            tests[key]["pvalue_adj"] = p_adj

    # Build matrices
    use_adj = adjust in {"holm"}
    pmat = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    dirmat = pd.DataFrame("", index=methods, columns=methods, dtype=str)
    ann = pd.DataFrame("", index=methods, columns=methods, dtype=str)

    for a_name, b_name, key in pair_keys:
        p_show = tests[key]["pvalue_adj"] if use_adj else tests[key]["pvalue_raw"]

        # Direction arrows (from the *row* perspective)
        # If A<B (A better), arrow "↑" when viewed as (row=A, col=B);
        # If A>B (B better), arrow "↓".
        if tests[key]["direction"] == "A<B":
            arrow_ab = "↑"
            arrow_ba = "↓"
        elif tests[key]["direction"] == "A>B":
            arrow_ab = "↓"
            arrow_ba = "↑"
        else:
            arrow_ab = arrow_ba = "→"

        pmat.loc[a_name, b_name] = p_show
        pmat.loc[b_name, a_name] = p_show
        dirmat.loc[a_name, b_name] = arrow_ab
        dirmat.loc[b_name, a_name] = arrow_ba

        # Annotated strings
        p_str = "NaN" if not (p_show == p_show) else f"{p_show:.6f}"
        ann.loc[a_name, b_name] = f"{arrow_ab} p={p_str}"
        ann.loc[b_name, a_name] = f"{arrow_ba} p={p_str}"

    np.fill_diagonal(pmat.values, np.nan)
    for m in methods:
        dirmat.loc[m, m] = ""
        ann.loc[m, m] = ""

    return tests, pmat, dirmat, ann


def save_summary_json(output_dir: str,
                      summary_df: pd.DataFrame,
                      wilcoxon_tests: dict,
                      filename: str = "summary.json"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    payload = {
        "summary": {
            method: {
                "mean_mse": float(summary_df.loc["mean", method]),
                "std_mse": float(summary_df.loc["std", method]),
            }
            for method in summary_df.columns
        },
        "wilcoxon": wilcoxon_tests,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Aggregated summary saved to {out_path}")

def save_summary_json(output_dir: str,
                      summary_df: pd.DataFrame,
                      wilcoxon_tests: dict,
                      filename: str = "summary.json"):
    """
    Saves summary and Wilcoxon test results to JSON.
    wilcoxon_tests should be the enriched dict from pairwise_wilcoxon().
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    payload = {
        "summary": {
            method: {
                "mean_mse": float(summary_df.loc["mean", method]),
                "std_mse": float(summary_df.loc["std", method]),
            }
            for method in summary_df.columns
        },
        "wilcoxon": wilcoxon_tests  # already has pvalue_raw, pvalue_adj, direction, win rates, etc.
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Aggregated summary saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize nested CV results and Wilcoxon tests")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Directory containing subfolders GCN, MLP, XGB, RF, SVM")
    parser.add_argument("--methods", type=str, nargs="*", default=["SAGE_n", "JMLP_tr"],#["GCN", "MLP", "XGB", "RF", "SVM"],
                        help="Subset/order of methods (default: GCN MLP XGB RF SVM)")
    parser.add_argument("--save_json", action="store_true",
                        help="If set, also writes summary.json to --output_dir")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write summary.json (required if --save_json)")
    args = parser.parse_args()

    summary_df, mse_dict = summarize_methods(args.root_dir, args.methods)
    print("\nMSE summary (rows: mean/std; columns: methods):")
    print(summary_df.to_string())

    tests, pvals, dirmat, annotated = pairwise_wilcoxon(mse_dict, adjust="holm")

    print("\nWilcoxon signed-rank p-values (Holm-adjusted):")
    print(pvals.to_string())

    print("\nDirection matrix (↑ row better, ↓ column better, → no diff):")
    print(dirmat.to_string())

    print("\nAnnotated matrix:")
    print(annotated.to_string())

    
    if args.save_json:
        if not args.output_dir:
            raise ValueError("--output_dir is required when --save_json is set")
        save_summary_json(args.output_dir, summary_df, tests, filename="summary.json")

if __name__ == "__main__":
    main()
