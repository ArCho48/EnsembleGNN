import os, pdb
import ast
import json
import time
import copy
import uuid
import pickle
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from deephyper.hpo import HpProblem, CBO
from deephyper.evaluator import Evaluator

# ---- local modules (unchanged) ----
from dataloader import (
    load_combined_pickle_data, create_graph_loader, create_ml_loader,
    load_pyg_data, get_ml_features
)
from models import (
    GCNNet, SAGENet, GATNet, GINNet, GINENet,
    PNANet, TransformerNet, DimeNetNet, SchNetNet,
    MLP, get_rf_model, get_svm_model, get_xgb_model
)
from evaluation import evaluate_and_plot
from hp_problem import define_hp_problem_graph, define_hp_problem_ml

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

# ---------------------- helpers for label standardization ----------------------
def _save_pickle(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _clone_graph(g):
    if hasattr(g, "clone"):
        return g.clone()
    return copy.deepcopy(g)

def _graph_targets_np(graphs):
    return np.array([float(g.y.item()) for g in graphs], dtype=np.float64)

def _fit_graph_y_scaler(train_graphs):
    ys = _graph_targets_np(train_graphs)
    mean = float(ys.mean())
    std = float(ys.std(ddof=0))
    if std < 1e-8:
        std = 1e-8
    return mean, std

def _apply_graph_y_standardize(graphs, mean, std):
    out = []
    for g in graphs:
        gc = _clone_graph(g)
        y = gc.y.float()
        y_std = (y - torch.tensor(mean, dtype=y.dtype, device=y.device)) / torch.tensor(std, dtype=y.dtype, device=y.device)
        gc.y = y_std.type_as(gc.y)
        out.append(gc)
    return out

def _inverse_standardize(arr, mean, std):
    return (np.asarray(arr, dtype=np.float64) * std) + mean

def _fit_array_y_scaler(y_train):
    ss = StandardScaler()
    y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    ss.fit(y_train_2d)
    return ss

def _transform_like(ss, y):
    orig_shape = y.shape
    y2 = ss.transform(y.reshape(-1, 1) if y.ndim == 1 else y)
    return y2.reshape(orig_shape)

def _inverse_transform_like(ss, y):
    orig_shape = y.shape
    y2 = ss.inverse_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
    return y2.reshape(orig_shape)

# ---------------------- splitting utilities ----------------------

def stratified_three_way_split_bins(
    y: np.ndarray, seed: int, train_frac=0.8, val_frac=0.1, test_frac=0.1, n_bins=10
):
    """Return indices for train/val/test with quantile-bin stratification on continuous y."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8, "Fractions must sum to 1.0"
    n = y.shape[0]
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_bins = kb.fit_transform(y.reshape(-1, 1)).ravel()

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_frac), random_state=seed)
    train_idx, valtest_idx = next(sss1.split(np.zeros(n), y_bins))

    # Second split on remaining: val vs test
    y_bins_valtest = y_bins[valtest_idx]
    val_size_rel = val_frac / (val_frac + test_frac + 1e-12)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_size_rel), random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(sss2.split(np.zeros_like(y_bins_valtest), y_bins_valtest))

    val_idx = valtest_idx[val_rel_idx]
    test_idx = valtest_idx[test_rel_idx]
    return train_idx, val_idx, test_idx

# ---------------------- objectives (train on train, validate on val, test on test) ----------------------

def objective_graph_single_split(hp_dict, model_name, tr_graphs_raw, va_graphs_raw, te_graphs_raw, save_root):
    """Train on standardized train, validate on standardized val, compute test MSE.
    Save the trained model and artifacts for this trial.
    """
    # per-trial directory
    trial_id = hp_dict.get("job_id","unknown")
    artifact_dir = Path(save_root) / f"{model_name}_trial_{trial_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # fit scaler on train only
    y_mean, y_std = _fit_graph_y_scaler(tr_graphs_raw)
    tr_graphs = _apply_graph_y_standardize(tr_graphs_raw, y_mean, y_std)
    va_graphs = _apply_graph_y_standardize(va_graphs_raw, y_mean, y_std)
    te_graphs = _apply_graph_y_standardize(te_graphs_raw, y_mean, y_std)

    bs = hp_dict.get('batch_size', 32)
    loader_tr = create_graph_loader(tr_graphs, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
    loader_va = create_graph_loader(va_graphs, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    loader_te = create_graph_loader(te_graphs, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

    in_dim = tr_graphs[0].x.size(1)
    model = GRAPH_MODELS[model_name](in_dim, hp_dict).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
    criterion = torch.nn.MSELoss()

    patience = hp_dict.get('patience', 10)
    min_delta = 1e-4
    best_val_mse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    val_losses = []

    for epoch in range(hp_dict['epochs']):
        model.train()
        for data in loader_tr:
            data = data.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), data.y.float())
            loss.backward(); optimizer.step()

        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for data in loader_va:
                data = data.to(DEVICE, non_blocking=True)
                preds.append(model(data).cpu().numpy())
                trues.append(data.y.cpu().numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        val_mse = np.mean((preds - trues) ** 2)

        # NEW: original-units val MSE and logging
        preds_va_orig = _inverse_standardize(preds, y_mean, y_std)
        trues_va_orig = _inverse_standardize(trues, y_mean, y_std)
        val_mse_orig = float(np.mean((preds_va_orig - trues_va_orig) ** 2))
        val_losses.append(val_mse_orig)

        if best_val_mse - val_mse > min_delta:
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    # test (standardized)
    preds_t, trues_t = [], []
    with torch.no_grad():
        for data in loader_te:
            data = data.to(DEVICE, non_blocking=True)
            preds_t.append(model(data).cpu().numpy())
            trues_t.append(data.y.cpu().numpy())
    preds_t = np.concatenate(preds_t); trues_t = np.concatenate(trues_t)
    test_mse_std = float(np.mean((preds_t - trues_t) ** 2))

    # original-units test MSE
    preds_orig = _inverse_standardize(preds_t, y_mean, y_std)
    trues_orig = _inverse_standardize(trues_t, y_mean, y_std)
    test_mse = float(np.mean((preds_orig - trues_orig) ** 2))

    # ---- SAVE ARTIFACTS ----
    # 1) model weights
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    # 2) hyperparams used in this trial
    with open(artifact_dir / "hps.json", "w") as f:
        json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in hp_dict.items()}, f, indent=2)
    # 3) y-scaler params needed for inference
    with open(artifact_dir / "y_scaler.json", "w") as f:
        json.dump({"mean": float(y_mean), "std": float(y_std)}, f, indent=2)
    # 4) quick metrics
    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump({
            "val_mse_std": float(best_val_mse),
            "test_mse_std": test_mse_std,
            "test_mse": test_mse,
            "best_epoch": int(best_epoch)
        }, f, indent=2)

    # parity plot in original units (optional but handy)
    evaluate_and_plot(
        y_true=trues_orig, y_pred=preds_orig,
        plot_path=str(artifact_dir / "parity.png"),
        metrics_path=str(artifact_dir / "metrics_dup.json")  # non-essential duplicate
    )

    # return neg val for HPO, include path to artifact dir
    return {
        "objective": -best_val_mse,
        "metadata": {
            "best_epoch": int(best_epoch),
            "val_mse_std": float(best_val_mse),
            "test_mse_std": test_mse_std,
            "test_mse": test_mse,
            "artifact_dir": str(artifact_dir),
            "val_mse_epochs": val_losses
        }
    }

def objective_ml_single_split(hp_dict, model_name, X_tr, y_tr, X_va, y_va, X_te, y_te, save_root):
    """Train on standardized train, validate on standardized val, compute test MSE.
    Save the trained model and artifacts for this trial.
    """
    # per-trial directory
    trial_id = hp_dict.get("job_id","unknown")
    artifact_dir = Path(save_root) / f"{model_name}_trial_{trial_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # y standardization on train only
    yss = _fit_array_y_scaler(y_tr)
    y_tr_s = _transform_like(yss, y_tr)
    y_va_s = _transform_like(yss, y_va)
    y_te_s = _transform_like(yss, y_te)

    if model_name == 'MLP':
        bs = hp_dict['batch_size']
        loader_tr = create_ml_loader(X_tr, y_tr_s, batch_size=bs)
        loader_va = create_ml_loader(X_va, y_va_s, batch_size=bs)
        loader_te = create_ml_loader(X_te, y_te_s, batch_size=bs, shuffle=False)

        model = MLP(X_tr.shape[1], hp_dict).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
        criterion = torch.nn.MSELoss()

        patience = hp_dict.get('patience', 5)
        min_delta = 1e-4
        best_val = float('inf')
        epochs_no_improve = 0

        for _ in range(hp_dict['epochs']):
            model.train()
            for xb, yb in loader_tr:
                xb = xb.to(DEVICE, non_blocking=True).float()
                yb = yb.to(DEVICE, non_blocking=True).float()
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward(); optimizer.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in loader_va:
                    xb = xb.to(DEVICE, non_blocking=True).float()
                    yb = yb.to(DEVICE, non_blocking=True).float()
                    out = model(xb)
                    preds.append(out.cpu().numpy()); trues.append(yb.cpu().numpy())
            preds = np.concatenate(preds); trues = np.concatenate(trues)
            val_mse = np.mean((preds - trues) ** 2)

            if best_val - val_mse > min_delta:
                best_val = val_mse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        # test (standardized)
        preds_t, trues_t = [], []
        with torch.no_grad():
            for xb, yb in loader_te:
                xb = xb.to(DEVICE, non_blocking=True).float()
                yb = yb.to(DEVICE, non_blocking=True).float()
                out = model(xb)
                preds_t.append(out.cpu().numpy()); trues_t.append(yb.cpu().numpy())
        preds_t = np.concatenate(preds_t); trues_t = np.concatenate(trues_t)
        test_mse_std = float(np.mean((preds_t - trues_t) ** 2))
        # original units
        preds_orig = _inverse_transform_like(yss, preds_t)
        trues_orig = _inverse_transform_like(yss, trues_t)
        test_mse = float(np.mean((preds_orig - trues_orig) ** 2))

        # ---- SAVE ARTIFACTS ----
        torch.save(model.state_dict(), artifact_dir / "model.pt")
        with open(artifact_dir / "hps.json", "w") as f:
            json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in hp_dict.items()}, f, indent=2)
        # save sklearn scaler
        _save_pickle(yss, artifact_dir / "y_scaler.pkl")
        with open(artifact_dir / "metrics.json", "w") as f:
            json.dump({"val_mse_std": float(best_val),
                       "test_mse_std": test_mse_std,
                       "test_mse": test_mse}, f, indent=2)
        # evaluate_and_plot(
        #     y_true=trues_orig, y_pred=preds_orig,
        #     plot_path=str(artifact_dir / "parity.png"),
        #     metrics_path=str(artifact_dir / "metrics_dup.json")
        # )

        return {"objective": -best_val,
                "metadata": {"val_mse_std": float(best_val),
                             "test_mse_std": test_mse_std,
                             "test_mse": test_mse,
                             "artifact_dir": str(artifact_dir)}}

    elif model_name == 'XGB':
        model = ML_MODELS[model_name](hp_dict)
        # ---- to numpy (CPU) + 1D targets ----
        def to_np(a):
            return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

        Xtr = to_np(X_tr); Xva = to_np(X_va); Xte = to_np(X_te)
        ytr = to_np(y_tr_s).reshape(-1)
        yva = to_np(y_va_s).reshape(-1)
        yte = to_np(y_te_s).reshape(-1)

        #  ---- set eval_metric on the model (NOT in fit) ----
        # use provided hp if any, else default to "rmse"
        try:
            if model.get_params().get("eval_metric", None) is None:
                model.set_params(eval_metric=hp_dict.get("eval_metric", "rmse"))
        except Exception:
            # tolerate older/odd wrappers
            pass

        # ---- early stopping: prefer fit kwarg, fallback to set_params ----
        esr = None
        try:
            esr = model.get_xgb_params().get("early_stopping_rounds", None)
        except Exception:
            pass
        if esr is None:
            esr = hp_dict.get("early_stopping_rounds", 50)

        fit_kwargs = {"eval_set": [(Xva, yva)], "early_stopping_rounds": esr}

        # Some versions don’t accept early_stopping_rounds in fit -> fallback
        try:
            model.fit(Xtr, ytr, **fit_kwargs)
        except TypeError:
            model.set_params(early_stopping_rounds=esr)
            model.fit(Xtr, ytr, eval_set=fit_kwargs["eval_set"])

        # ---- metrics on standardized scale ----
        preds_val = model.predict(Xva)
        val_mse = float(np.mean((preds_val - yva) ** 2))

        preds_t = model.predict(Xte)
        test_mse_std = float(np.mean((preds_t - yte) ** 2))

        # ---- back to original units ----
        preds_orig = _inverse_transform_like(yss, preds_t)
        y_te_np = to_np(y_te).reshape(-1)  # original-scale y
        test_mse = float(np.mean((preds_orig - y_te_np) ** 2))

        # ---- SAVE ARTIFACTS ----
        _save_pickle(model, artifact_dir / "model.pkl")
        with open(artifact_dir / "hps.json", "w") as f:
            json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in hp_dict.items()}, f, indent=2)
        _save_pickle(yss, artifact_dir / "y_scaler.pkl")
        with open(artifact_dir / "metrics.json", "w") as f:
            json.dump({"val_mse_std": val_mse,
                       "test_mse_std": test_mse_std,
                       "test_mse": test_mse}, f, indent=2)

        return {"objective": -val_mse,
                "metadata": {"val_mse_std": val_mse,
                             "test_mse_std": test_mse_std,
                             "test_mse": test_mse,
                             "artifact_dir": str(artifact_dir)}}

    else:  # RF, SVM, etc.
        model = ML_MODELS[model_name](hp_dict)
        model.fit(X_tr.numpy(), y_tr_s.numpy() if hasattr(y_tr_s, "numpy") else y_tr_s)

        preds_val = model.predict(X_va.numpy())
        val_mse = float(np.mean((preds_val - (y_va_s if isinstance(y_va_s, np.ndarray) else y_va_s.numpy())) ** 2))

        preds_t = model.predict(X_te.numpy())
        test_mse_std = float(np.mean((preds_t - (y_te_s if isinstance(y_te_s, np.ndarray) else y_te_s.numpy())) ** 2))
        preds_orig = _inverse_transform_like(yss, preds_t)
        y_te_np = y_te.numpy() if isinstance(y_te, torch.Tensor) else np.asarray(y_te)
        test_mse = float(np.mean((preds_orig - y_te_np) ** 2))

        # ---- SAVE ARTIFACTS ----
        _save_pickle(model, artifact_dir / "model.pkl")
        with open(artifact_dir / "hps.json", "w") as f:
            json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in hp_dict.items()}, f, indent=2)
        _save_pickle(yss, artifact_dir / "y_scaler.pkl")
        with open(artifact_dir / "metrics.json", "w") as f:
            json.dump({"val_mse_std": val_mse,
                       "test_mse_std": test_mse_std,
                       "test_mse": test_mse}, f, indent=2)

        return {"objective": -val_mse,
                "metadata": {"val_mse_std": val_mse,
                             "test_mse_std": test_mse_std,
                             "test_mse": test_mse,
                             "artifact_dir": str(artifact_dir)}}

# ---------------------- main HPO routine (no CV, no retraining) ----------------------
def _to_numpy_float64_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().view(-1).numpy()
    else:
        x = np.asarray(x).reshape(-1)
    return x.astype(np.float64)

def run_hpo_single_split(model_name, train_graphs, features, labels, args):
    # pdb.set_trace()
    # ---- 0.8 / 0.1 / 0.1 stratified split ----
    if model_name in GRAPH_MODELS:
        y = np.array([g.y.item() for g in train_graphs], dtype=np.float64)
    else:
        y = _to_numpy_float64_1d(labels)
    
    tr_idx, va_idx, te_idx = stratified_three_way_split_bins(
        y=y, seed=args.seed, train_frac=0.8, val_frac=0.1, test_frac=0.1, n_bins=args.strat_bins
    )
    
    # ---- per-model directories ----
    model_root = os.path.join(args.model_dir, 'hpo_single_no_retrain', model_name)
    out_root = os.path.join(args.output_dir, 'hpo_single_no_retrain', model_name)
    os.makedirs(model_root, exist_ok=True); os.makedirs(out_root, exist_ok=True)

    # where each trial will be saved
    trials_root = os.path.join(out_root, "trials_raw")
    os.makedirs(trials_root, exist_ok=True)

    # ---- define HpProblem & objective closure ----
    if model_name in GRAPH_MODELS:
        hp_problem = define_hp_problem_graph(model_name)
        tr_graphs_raw = [train_graphs[i] for i in tr_idx]
        va_graphs_raw = [train_graphs[i] for i in va_idx]
        te_graphs_raw = [train_graphs[i] for i in te_idx]

        def run_once(hp_config):
            return objective_graph_single_split(
                hp_config, model_name,
                tr_graphs_raw, va_graphs_raw, te_graphs_raw,
                save_root=trials_root
            )

    else:
        hp_problem = define_hp_problem_ml(model_name)
        X_tr, X_va, X_te = features[tr_idx], features[va_idx], features[te_idx]
        y_tr, y_va, y_te = labels[tr_idx], labels[va_idx], labels[te_idx]

        def run_once(hp_config):
            return objective_ml_single_split(
                hp_config, model_name,
                X_tr, y_tr, X_va, y_va, X_te, y_te,
                save_root=trials_root
            )

    # ---- DeepHyper search ----
    evaluator = Evaluator.create(
        run_function=run_once,
        method="ray",
        method_kwargs={
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
            "num_cpus_per_task": 1,
            "num_gpus_per_task": 1,
        },
    )
    search = CBO(
        hp_problem, evaluator,
        initial_points=[hp_problem.default_configuration],
        acq_optimizer='mixedga',
        log_dir=model_root
    )
    history = search.search(max_evals=args.max_evals)

    # Persist full search table
    history_csv = os.path.join(out_root, "search_history.csv")
    try:
        history.to_csv(history_csv, index=False)
    except Exception:
        pass

    # Extract per-trial metrics quickly (val/test)
    trials = []
    hist = history.reset_index(drop=True)
    for i, row in hist.iterrows():
        # hyperparams
        hps = {}
        for col in history.columns:
            if col.startswith('p:'):
                key = col.split('p:')[1]
                raw_val = row[col]
                if isinstance(raw_val, str):
                    try:
                        val = ast.literal_eval(raw_val)
                    except Exception:
                        val = raw_val
                else:
                    val = raw_val
                if isinstance(val, np.generic):
                    val = val.item()
                hps[key] = val

        # metadata fields if present
        meta = {}
        for col in history.columns:
            if col.startswith('m:'):
                k = col.split('m:', 1)[1]
                v = row[col]
                if isinstance(v, np.generic):
                    v = v.item()
                meta[k] = v

        trials.append({
            "rank": i,
            "objective": float(row.get("objective", np.nan)),
            "val_mse_std": float(meta.get("val_mse_std", np.nan)),
            "test_mse_std": float(meta.get("test_mse_std", np.nan)),
            "test_mse": float(meta.get("test_mse", np.nan)),
            "best_epoch": int(meta.get("best_epoch", -1)) if "best_epoch" in meta else None,
            "artifact_dir": meta.get("artifact_dir", None),
            "hps": hps
        })

    with open(os.path.join(out_root, "trials.json"), "w") as f:
        json.dump(trials, f, indent=2)

    # also write a tiny summary file for the best-by-val trial
    best_idx = history['objective'].idxmax()
    best_row = history.loc[best_idx]
    best_summary = {
        "objective": float(best_row["objective"]),
        "val_mse_std": float(best_row.get("m:val_mse_std", np.nan)),
        "test_mse_std": float(best_row.get("m:test_mse_std", np.nan)),
        "test_mse": float(best_row.get("m:test_mse", np.nan)),
        "artifact_dir": best_row.get("m:artifact_dir", None)
    }
    with open(os.path.join(out_root, "best_summary.json"), "w") as f:
        json.dump(best_summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='HPO with single 0.8/0.1/0.1 stratified split. Saves every trained model.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--models', nargs='+', required=True,
                        help='Choose from graph: {} or ML: {}'.format(list(GRAPH_MODELS.keys()), list(ML_MODELS.keys())))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # HPO resources
    parser.add_argument('--max_evals', type=int, default=1200)
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=4)

    # Stratification bins over y
    parser.add_argument('--strat_bins', type=int, default=10)

    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_graphs = load_pyg_data(args.data_dir)
    features, labels = get_ml_features(train_graphs)

    for model_name in args.models:
        print(f"[HPO] {model_name} ...")
        run_hpo_single_split(model_name, train_graphs, features, labels, args)
    print("HPO complete. Per-trial VAL/TEST metrics dumped and all trained models saved.")

if __name__ == '__main__':
    main()
