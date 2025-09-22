import os
import ast
import json
import math
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer

# ---------- Local project imports (same as your HPO script) ----------
from dataloader import (
    load_combined_pickle_data, create_graph_loader, create_ml_loader,
    load_pyg_data, get_ml_features
)
from models import (
    GCNNet, SAGENet, GATNet, GINNet, GINENet,
    PNANet, TransformerNet, DimeNetNet, SchNetNet,
    MLP, get_rf_model, get_svm_model, get_xgb_model
)

# ----------------- Constants & registries -----------------
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRAPH_MODELS = {
    'GCN': GCNNet,
    'SAGE': SAGENet,
    'GAT': GATNet,
    'GIN': GINNet,
    'GINE': GINENet,
    'PNA': PNANet,
    'Transformer': TransformerNet,
    'DimeNet': DimeNetNet,
    'SchNet': SchNetNet,
}
ML_MODELS = {
    'MLP': MLP,
    'RF': get_rf_model,
    'SVM': get_svm_model,
    'XGB': get_xgb_model,
}

# ----------------- Split helper (identical to HPO) -----------------
def stratified_three_way_split_bins(
    y: np.ndarray, seed: int, train_frac=0.8, val_frac=0.1, test_frac=0.1, n_bins=10
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8
    n = y.shape[0]
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_bins = kb.fit_transform(y.reshape(-1, 1)).ravel()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_frac), random_state=seed)
    train_idx, valtest_idx = next(sss1.split(np.zeros(n), y_bins))

    y_bins_valtest = y_bins[valtest_idx]
    val_size_rel = val_frac / (val_frac + test_frac + 1e-12)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_size_rel), random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(sss2.split(np.zeros_like(y_bins_valtest), y_bins_valtest))

    val_idx = valtest_idx[val_rel_idx]
    test_idx = valtest_idx[test_rel_idx]
    return train_idx, val_idx, test_idx

# ----------------- Utils -----------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _load_history(history_path: str) -> pd.DataFrame:
    df = pd.read_csv(history_path)
    return df

def _find_trials_root(model_out_root: str) -> str:
    troot = os.path.join(model_out_root, "trials_raw")
    if not os.path.isdir(troot):
        raise FileNotFoundError(f"Trials directory not found: {troot}")
    return troot

def _get_artifact_dir_from_row(row: pd.Series) -> str:
    for key in row.index:
        if key.startswith("m:") and "artifact_dir" in key:
            return row[key]
    return None

# ----------------- Prediction loaders per family -----------------
def load_graph_artifacts(artifact_dir: str) -> Dict:
    artifact = Path(artifact_dir)
    with open(artifact / "hps.json") as f:
        hps = json.load(f)
    with open(artifact / "y_scaler.json") as f:
        ysc = json.load(f)  # {"mean":..., "std":...}
    state_path = artifact / "model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {artifact_dir}")
    return {"hps": hps, "y_scaler": ysc, "state_dict_path": str(state_path)}

def load_ml_artifacts(artifact_dir: str, model_name: str) -> Dict:
    artifact = Path(artifact_dir)
    with open(artifact / "hps.json") as f:
        hps = json.load(f)
    y_scaler_pkl = artifact / "y_scaler.pkl"
    if not y_scaler_pkl.exists():
        raise FileNotFoundError(f"Missing y_scaler.pkl in {artifact_dir}")
    yss = pickle.load(open(y_scaler_pkl, "rb"))

    if model_name == "MLP":
        state_path = artifact / "model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing model.pt in {artifact_dir}")
        return {"hps": hps, "y_scaler": yss, "state_dict_path": str(state_path)}
    else:
        model_pkl = artifact / "model.pkl"
        if not model_pkl.exists():
            raise FileNotFoundError(f"Missing model.pkl in {artifact_dir}")
        model = pickle.load(open(model_pkl, "rb"))
        return {"hps": hps, "y_scaler": yss, "model": model}

# ----------------- Build predictions (original units) -----------------
@torch.no_grad()
def predict_graph(model_name: str,
                  hps: Dict,
                  state_dict_path: str,
                  graphs: List,
                  y_scaler: Dict,
                  batch_size: int = None) -> np.ndarray:
    bs = int(hps.get("batch_size", 32)) if batch_size is None else batch_size
    loader = create_graph_loader(graphs, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

    in_dim = graphs[0].x.size(1)
    net = GRAPH_MODELS[model_name](in_dim, hps).to(DEVICE)
    net.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
    net.eval()

    preds_std = []
    for data in loader:
        data = data.to(DEVICE, non_blocking=True)
        out = net(data).detach().cpu().numpy()
        preds_std.append(out)
    preds_std = np.concatenate(preds_std, axis=0).reshape(-1)

    mean, std = float(y_scaler["mean"]), float(y_scaler["std"])
    preds = preds_std * std + mean
    return preds

@torch.no_grad()
def predict_mlp(hps: Dict,
                state_dict_path: str,
                X: np.ndarray,
                y_scaler,
                batch_size: int = None) -> np.ndarray:
    bs = int(hps.get("batch_size", 64)) if batch_size is None else batch_size
    loader = create_ml_loader(X, np.zeros((X.shape[0], 1), dtype=np.float32), batch_size=bs, shuffle=False)
    net = MLP(X.shape[1], hps).to(DEVICE)
    net.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
    net.eval()

    preds_std = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        out = net(xb).detach().cpu().numpy()
        preds_std.append(out)
    preds_std = np.concatenate(preds_std, axis=0).reshape(-1)

    preds = y_scaler.inverse_transform(preds_std.reshape(-1, 1)).reshape(-1)
    return preds

def predict_sklearn(model, X: np.ndarray, y_scaler) -> np.ndarray:
    preds_std = model.predict(X)
    preds = y_scaler.inverse_transform(np.array(preds_std).reshape(-1, 1)).reshape(-1)
    return preds

# ----------------- Plotting (with 95% CI) -----------------
def _parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str):
    """Parity scatter with y=x line and a global 95% CI band from residual std."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, s=16, alpha=0.6)

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    xx = np.linspace(mn, mx, 200)

    # global 95% CI from residual std (homoscedastic approximation)
    resid = y_pred - y_true
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    ci = 1.96 * sigma
    plt.plot(xx, xx, linestyle='--')
    plt.fill_between(xx, xx - ci, xx + ci, alpha=0.15, linewidth=0, label='95% CI (global)')

    plt.xlabel('$\\Delta G_{ox}$ (eV) - Actual Values', fontsize=20)
    plt.ylabel('$\\Delta G_{ox}$ (eV) - Predicted Values', fontsize=20)
    plt.title(title)
    plt.legend(loc='best', frameon=False, fontsize=18)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _parity_plot_with_var(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          var: np.ndarray,
                          out_path: str,
                          title: str,
                          n_members: int = None):
    """
    Parity scatter with y=x and 95% CI error bars.
    var: per-sample variance across ensemble members.
    If n_members is provided, CI for the ensemble mean: 1.96 * sqrt(var / n_members)
    else: 1.96 * sqrt(var) (conservative).
    """
    var = np.asarray(var)
    if n_members is not None and n_members > 1:
        std_err = np.sqrt(np.clip(var, 0.0, None) / float(n_members))
    else:
        std_err = np.sqrt(np.clip(var, 0.0, None))
    ci_y = 1.96 * std_err

    plt.figure(figsize=(8, 6))
    plt.errorbar(y_true, y_pred, yerr=ci_y, fmt='o', markersize=4, alpha=0.55,
                 ecolor='gray', elinewidth=1, capsize=2, label='95% CI (per-point)')

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    xx = np.linspace(mn, mx, 200)
    plt.plot(xx, xx, linestyle='--')

    # also add a global CI band based on residuals (optional)
    resid = y_pred - y_true
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    ci_global = 1.96 * sigma
    plt.fill_between(xx, xx - ci_global, xx + ci_global, alpha=0.12, linewidth=0, label='95% CI (global)')

    plt.xlabel('$\\Delta G_{ox}$ (eV) - Actual Values', fontsize=20)
    plt.ylabel('$\\Delta G_{ox}$ (eV) - Predicted Values', fontsize=20)
    plt.title(title)
    plt.legend(loc='best', frameon=False, fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------- Ensemble utilities -----------------
def topk_mean_ensemble(preds_matrix: np.ndarray, k: int, order_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean_pred, member_preds) where:
      - mean_pred: [N] averaged predictions of top-k (by order)
      - member_preds: [k, N] the member predictions used
    """
    idx = order_idx[:k]
    member_preds = preds_matrix[idx]  # [k, N]
    mean_pred = member_preds.mean(axis=0)
    return mean_pred, member_preds

def greedy_forward_ensemble(val_true: np.ndarray,
                            preds_val_matrix: np.ndarray,
                            order_idx: np.ndarray,
                            max_k: int) -> List[int]:
    """
    Greedy forward selection on validation RMSE.
    Returns chosen indices (subset of order_idx) in selection order.
    """
    chosen = []
    current_pred = None
    best_rmse = math.inf

    for i in order_idx:
        if len(chosen) >= max_k:
            break
        candidate_preds = preds_val_matrix[i]
        if current_pred is None:
            tmp = candidate_preds
        else:
            tmp = (current_pred * len(chosen) + candidate_preds) / (len(chosen) + 1)
        cur_rmse = rmse(val_true, tmp)
        if cur_rmse + 1e-12 < best_rmse:
            best_rmse = cur_rmse
            chosen.append(i)
            current_pred = tmp
    return chosen

def ensemble_variance(member_preds: np.ndarray) -> np.ndarray:
    """
    Epistemic uncertainty via ensemble variance across members.
    member_preds: [K, N]
    Returns per-sample variance: [N]
    """
    return member_preds.var(axis=0, ddof=1) if member_preds.shape[0] > 1 else np.zeros(member_preds.shape[1])

def _to_numpy_float64_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().view(-1).numpy()
    else:
        x = np.asarray(x).reshape(-1)
    return x.astype(np.float64)

# ----------------- Main routine -----------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate default/best/topK/greedy ensembles on the fixed HPO test set, with progress logs, epistemic uncertainty, and parity plots (95% CI).")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, help=f"One of {list(GRAPH_MODELS.keys()) + list(ML_MODELS.keys())}")
    parser.add_argument("--output_dir", type=str, required=True, help="Same --output_dir used in HPO script")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strat_bins", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    if args.device == "cuda" and torch.cuda.is_available():
        global DEVICE
        DEVICE = torch.device("cuda")

    # ---------- Locate HPO outputs ----------
    model_out_root = os.path.join(args.output_dir, "hpo_single_no_retrain", args.model_name)
    history_path = os.path.join(model_out_root, "search_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Cannot find history csv at {history_path}")
    trials_root = _find_trials_root(model_out_root)

    print("[INFO] Loading history:", history_path)
    hist = _load_history(history_path)

    # ---------- Load data & rebuild same split ----------
    print("[INFO] Loading data and rebuilding split...")
    train_graphs = load_pyg_data(args.data_dir)
    features, labels = get_ml_features(train_graphs)

    if args.model_name in GRAPH_MODELS:
        y_all = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float64)
    else:
        y_all = _to_numpy_float64_1d(labels)

    tr_idx, va_idx, te_idx = stratified_three_way_split_bins(
        y=y_all, seed=args.seed, train_frac=0.8, val_frac=0.1, test_frac=0.1, n_bins=args.strat_bins
    )

    if args.model_name in GRAPH_MODELS:
        tr_graphs = [train_graphs[i] for i in tr_idx]
        va_graphs = [train_graphs[i] for i in va_idx]
        te_graphs = [train_graphs[i] for i in te_idx]
        y_val_true = np.array([float(g.y.item()) for g in va_graphs], dtype=np.float64)
        y_test_true = np.array([float(g.y.item()) for g in te_graphs], dtype=np.float64)
        print(f"[INFO] Graph data: train={len(tr_graphs)}, val={len(va_graphs)}, test={len(te_graphs)}")
    else:
        X_tr, X_va, X_te = features[tr_idx], features[va_idx], features[te_idx]
        y_tr, y_va, y_te = labels[tr_idx], labels[va_idx], labels[te_idx]
        y_val_true = _to_numpy_float64_1d(y_va).reshape(-1)
        y_test_true = _to_numpy_float64_1d(y_te).reshape(-1)
        print(f"[INFO] ML data: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")

    # ---------- Normalize history columns ----------
    print("[INFO] Normalizing history columns...")
    if "job_id" not in hist.columns:
        jcols = [c for c in hist.columns if c.lower() in ("job_id", "jobid", "job")]
        if jcols:
            hist = hist.rename(columns={jcols[0]: "job_id"})
        else:
            raise KeyError("Could not find a job_id column in history CSV.")

    if "m:val_mse_std" not in hist.columns:
        alt = [c for c in hist.columns if c.startswith("m:") and "val" in c and "mse" in c]
        if alt:
            hist = hist.rename(columns={alt[0]: "m:val_mse_std"})
        else:
            raise KeyError("Could not find 'm:val_mse_std' in history.")

    if "m:artifact_dir" not in hist.columns:
        trials_json = os.path.join(model_out_root, "trials.json")
        if os.path.exists(trials_json):
            tj = json.load(open(trials_json))
            ad_map = {i: t.get("artifact_dir") for i, t in enumerate(tj)}
            hist["m:artifact_dir"] = hist.index.map(ad_map.get)
        else:
            raise KeyError("History missing 'm:artifact_dir' and no trials.json to recover it.")

    # ---------- Identify default and best models ----------
    print("[INFO] Selecting default (job_id==0) and best (min val MSE) models...")
    default_rows = hist[hist["job_id"] == 0]
    if default_rows.empty:
        raise ValueError("No row with job_id==0 found for default model.")
    default_row = default_rows.iloc[0]
    best_idx = hist["m:val_mse_std"].astype(float).idxmin()
    best_row = hist.loc[best_idx]
    order = hist["m:val_mse_std"].astype(float).sort_values().index.values  # ascending by val MSE

    # ---------- Compute preds for ALL models (val/test) ----------
    print("[INFO] Loading artifacts and computing predictions for ALL trials...")
    preds_val_list, preds_test_list, art_dirs = [], [], []
    for ridx in hist.index:
        row = hist.loc[ridx]
        art_dir = _get_artifact_dir_from_row(row)
        if not art_dir or not os.path.isdir(art_dir):
            cand = os.path.join(trials_root, os.path.basename(str(art_dir))) if art_dir else None
            if cand and os.path.isdir(cand):
                art_dir = cand
            else:
                raise FileNotFoundError(f"artifact_dir not found for row {ridx}: {art_dir}")

        if args.model_name in GRAPH_MODELS:
            art = load_graph_artifacts(art_dir)
            preds_val = predict_graph(args.model_name, art["hps"], art["state_dict_path"], va_graphs, art["y_scaler"])
            preds_test = predict_graph(args.model_name, art["hps"], art["state_dict_path"], te_graphs, art["y_scaler"])
        else:
            art = load_ml_artifacts(art_dir, args.model_name)
            if args.model_name == "MLP":
                preds_val = predict_mlp(art["hps"], art["state_dict_path"], X_va, art["y_scaler"])
                preds_test = predict_mlp(art["hps"], art["state_dict_path"], X_te, art["y_scaler"])
            else:
                preds_val = predict_sklearn(art["model"], X_va, art["y_scaler"])
                preds_test = predict_sklearn(art["model"], X_te, art["y_scaler"])

        preds_val_list.append(preds_val.reshape(-1))
        preds_test_list.append(preds_test.reshape(-1))
        art_dirs.append(art_dir)

        if len(preds_val_list) % 50 == 0:
            print(f"[INFO] Processed {len(preds_val_list)} / {len(hist)} trials...")

    preds_val = np.stack(preds_val_list, axis=0)   # [M, N_val]
    preds_test = np.stack(preds_test_list, axis=0) # [M, N_test]
    print("[INFO] Finished computing predictions.")

    # Mapping from history row index -> position in stacked arrays
    row_to_pos = {idx: pos for pos, idx in enumerate(hist.index.values)}

    # ---------- (1) Default model RMSE + parity (95% CI band) ----------
    print("[INFO] Evaluating DEFAULT model (job_id==0)...")
    d_idx = default_row.name
    default_pos = row_to_pos[d_idx]
    y_pred_default = preds_test[default_pos]
    rmse_default = rmse(y_test_true, y_pred_default)
    _parity_plot(y_test_true, y_pred_default, os.path.join(model_out_root, "parity_default.png"),
                 f"Default (job_id==0) — RMSE={rmse_default:.4f}")

    # ---------- (2) Best single model by val RMSE + parity (95% CI band) ----------
    print("[INFO] Evaluating BEST single model (by lowest validation MSE)...")
    best_pos = row_to_pos[best_idx]
    y_pred_best = preds_test[best_pos]
    rmse_best = rmse(y_test_true, y_pred_best)
    _parity_plot(y_test_true, y_pred_best, os.path.join(model_out_root, "parity_best.png"),
                 f"Best Single (min val MSE) — RMSE={rmse_best:.4f}")

    # ---------- (3) Top-K mean ensemble + parity with per-point 95% CI ----------
    K = min(args.top_k, preds_val.shape[0])
    print(f"[INFO] Building TOP-K ensemble (K={K})...")
    order_pos = np.array([row_to_pos[i] for i in order], dtype=int)
    ens_topk_val_mean, topk_member_val = topk_mean_ensemble(preds_val, K, order_pos)
    ens_topk_test_mean, topk_member_test = topk_mean_ensemble(preds_test, K, order_pos)

    rmse_topk_val = rmse(y_val_true, ens_topk_val_mean)
    rmse_topk_test = rmse(y_test_true, ens_topk_test_mean)

    print("[INFO] Computing epistemic uncertainty (variance) for TOP-K ensemble...")
    var_topk_val = ensemble_variance(topk_member_val)   # [N_val]
    var_topk_test = ensemble_variance(topk_member_test) # [N_test]
    var_topk_val_mean = float(var_topk_val.mean())
    var_topk_test_mean = float(var_topk_test.mean())

    _parity_plot_with_var(y_test_true, ens_topk_test_mean, var_topk_test,
                          os.path.join(model_out_root, "parity_topk.png"),
                          f"Top-{K} Ensemble — RMSE={rmse_topk_test:.4f} (mean var={var_topk_test_mean:.3g})",
                          n_members=K)

    # ---------- (4) Greedy forward ensemble (on val) + parity with per-point 95% CI ----------
    print("[INFO] Running GREEDY forward selection on validation set...")
    chosen_pos = greedy_forward_ensemble(y_val_true, preds_val, order_pos, max_k=K)
    print(f"[INFO] Greedy selected {len(chosen_pos)} members.")
    if len(chosen_pos) == 0:
        chosen_pos = [best_pos]

    greedy_member_val = preds_val[chosen_pos]     # [G, N_val]
    greedy_member_test = preds_test[chosen_pos]   # [G, N_test]
    ens_greedy_val_mean = greedy_member_val.mean(axis=0)
    ens_greedy_test_mean = greedy_member_test.mean(axis=0)

    rmse_greedy_val = rmse(y_val_true, ens_greedy_val_mean)
    rmse_greedy_test = rmse(y_test_true, ens_greedy_test_mean)

    print("[INFO] Computing epistemic uncertainty (variance) for GREEDY ensemble...")
    var_greedy_val = ensemble_variance(greedy_member_val)
    var_greedy_test = ensemble_variance(greedy_member_test)
    var_greedy_val_mean = float(var_greedy_val.mean())
    var_greedy_test_mean = float(var_greedy_test.mean())

    _parity_plot_with_var(y_test_true, ens_greedy_test_mean, var_greedy_test,
                          os.path.join(model_out_root, "parity_greedy.png"),
                          f"Greedy Ensemble — RMSE={rmse_greedy_test:.4f} (mean var={var_greedy_test_mean:.3g})",
                          n_members=len(chosen_pos))

    # ---------- Print & save summary ----------
    summary = {
        "model_name": args.model_name,
        "num_models": int(preds_val.shape[0]),
        "top_k_used": int(K),

        "rmse_default_test": rmse_default,
        "rmse_best_single_test": rmse_best,
        "rmse_topk_test": rmse_topk_test,
        "rmse_greedy_test": rmse_greedy_test,

        "rmse_topk_val": rmse_topk_val,
        "rmse_greedy_val": rmse_greedy_val,

        # Epistemic uncertainty (mean variance across samples)
        "var_topk_val_mean": var_topk_val_mean,
        "var_topk_test_mean": var_topk_test_mean,
        "var_greedy_val_mean": var_greedy_val_mean,
        "var_greedy_test_mean": var_greedy_test_mean,
    }

    print("\n================= RESULTS =================")
    print(json.dumps({
        "Default (job_id==0) RMSE (test)": summary["rmse_default_test"],
        "Best single RMSE (test)": summary["rmse_best_single_test"],
        "TopK ensemble RMSE (test)": summary["rmse_topk_test"],
        "Greedy ensemble RMSE (test)": summary["rmse_greedy_test"],
        "TopK epistemic var (test, mean)": summary["var_topk_test_mean"],
        "Greedy epistemic var (test, mean)": summary["var_greedy_test_mean"],
    }, indent=2))
    print("===========================================\n")

    out_dir = model_out_root
    out_json = os.path.join(out_dir, "ensemble_eval_summary.json")
    print("[INFO] Saving summary to:", out_json)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Save the full per-sample variance vectors for post-hoc analysis
    print("[INFO] Saving per-sample variance arrays...")
    np.save(os.path.join(out_dir, "var_topk_val.npy"), var_topk_val)
    np.save(os.path.join(out_dir, "var_topk_test.npy"), var_topk_test)
    np.save(os.path.join(out_dir, "var_greedy_val.npy"), var_greedy_val)
    np.save(os.path.join(out_dir, "var_greedy_test.npy"), var_greedy_test)

    # Also handy to save which members were used
    np.save(os.path.join(out_dir, "greedy_selected_indices.npy"), np.array(chosen_pos, dtype=int))
    np.save(os.path.join(out_dir, "topk_selected_indices.npy"), np.array(order_pos[:K], dtype=int))

    print("[INFO] Saved parity plots to:")
    print("   -", os.path.join(out_dir, "parity_default.png"))
    print("   -", os.path.join(out_dir, "parity_best.png"))
    print("   -", os.path.join(out_dir, "parity_topk.png"))
    print("   -", os.path.join(out_dir, "parity_greedy.png"))
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
