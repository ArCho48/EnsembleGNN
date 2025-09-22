import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import pdb
import ast
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.benchmark = True
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from scipy.stats import wilcoxon
import copy

from deephyper.hpo import HpProblem, CBO
from deephyper.evaluator import Evaluator

from dataloader import ( load_combined_pickle_data, create_graph_loader, 
                        create_ml_loader, load_pyg_data, get_ml_features )
from models import (
    GCNNet, SAGENet, GATNet, GINNet, GINENet,
    PNANet, TransformerNet, DimeNetNet, SchNetNet,
    MLP, get_rf_model, get_svm_model, get_xgb_model
)
from evaluation import evaluate_and_plot
from hp_problem import define_hp_problem_graph, define_hp_problem_ml

# Map model names to constructors and run-functions
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- helpers for label standardization ----------------------

def _clone_graph(g):
    # Safe clone without requiring torch_geometric import in this file
    if hasattr(g, "clone"):
        return g.clone()
    return copy.deepcopy(g)

def _graph_targets_np(graphs):
    """Stack scalar targets from a list of graphs -> (N,) float64"""
    return np.array([float(g.y.item()) for g in graphs], dtype=np.float64)

def _fit_graph_y_scaler(train_graphs):
    """Fit mean/std from train graphs' y (scalar) and return (mean, std) as floats with std>=1e-8."""
    ys = _graph_targets_np(train_graphs)
    mean = float(ys.mean())
    std = float(ys.std(ddof=0))
    if std < 1e-8:
        std = 1e-8
    return mean, std

def _apply_graph_y_standardize(graphs, mean, std):
    """Return new list of graphs with y standardized: (y - mean) / std."""
    out = []
    for g in graphs:
        gc = _clone_graph(g)
        # ensure float tensor
        y = gc.y.float()
        # handle scalar or vector (but your code uses scalar y)
        y_std = (y - torch.tensor(mean, dtype=y.dtype, device=y.device)) / torch.tensor(std, dtype=y.dtype, device=y.device)
        gc.y = y_std.type_as(gc.y)
        out.append(gc)
    return out

def _inverse_standardize(arr, mean, std):
    """Inverse for numpy arrays: arr * std + mean."""
    return (np.asarray(arr, dtype=np.float64) * std) + mean

def _fit_array_y_scaler(y_train):
    """Return sklearn StandardScaler fit on y_train (any shape), and a function to transform preserving shape."""
    ss = StandardScaler()
    y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    ss.fit(y_train_2d)
    return ss

def _transform_like(ss, y):
    """Apply ss.transform to y and return with original shape preserved."""
    orig_shape = y.shape
    y2 = ss.transform(y.reshape(-1, 1) if y.ndim == 1 else y)
    return y2.reshape(orig_shape)

def _inverse_transform_like(ss, y):
    orig_shape = y.shape
    y2 = ss.inverse_transform(y.reshape(-1, 1) if y.ndim == 1 else y)
    return y2.reshape(orig_shape)

# ------------------------------------------------------------------------------

def inner_objective_graph(hp_dict, model_name, train_graphs, inner_folds, seed):
    """
    Objective for graph models: average neg-MSE across inner folds.
    Targets (data.y) are standardized per inner-train fold using mean/std from that fold.
    """
    ys = np.array([data.y.item() for data in train_graphs])
    # Stratify on quantile bins
    kb = KBinsDiscretizer(n_bins=inner_folds, encode='ordinal', strategy='quantile')
    y_bins = kb.fit_transform(ys.reshape(-1,1)).flatten()
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    mses = []
    best_epochs = []
    for train_idx, val_idx in skf.split(train_graphs, y_bins):
        # fit scaler on inner-train and standardize y for tr/va
        tr_raw = [train_graphs[i] for i in train_idx]
        va_raw = [train_graphs[i] for i in val_idx]
        y_mean, y_std = _fit_graph_y_scaler(tr_raw)
        tr = _apply_graph_y_standardize(tr_raw, y_mean, y_std)
        va = _apply_graph_y_standardize(va_raw, y_mean, y_std)

        tr_loader = create_graph_loader(tr, batch_size=hp_dict.get('batch_size',32), shuffle=True, num_workers=8, pin_memory=True)
        va_loader = create_graph_loader(va, batch_size=hp_dict.get('batch_size',32), shuffle=False, num_workers=8, pin_memory=True)

        # Build fresh model on device for the inner fold
        if torch.device(DEVICE).type == "cuda":
            with torch.device(DEVICE):
                model = GRAPH_MODELS[model_name](train_graphs[0].x.size(1), hp_dict)
        else:
            model = GRAPH_MODELS[model_name](train_graphs[0].x.size(1), hp_dict).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
        criterion = torch.nn.MSELoss()
        # Early‐stopping settings
        patience = hp_dict.get('patience', 10)   
        min_delta = 1e-4                        
        best_fold_mse = float('inf')
        best_epoch = 1
        epochs_no_improve = 0
        # train for hp_dict['epochs']
        for epoch in range(hp_dict['epochs']):
            model.train()
            for data in tr_loader:
                data = data.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data), data.y.float())  # y is standardized
                loss.backward(); optimizer.step()
            # eval on standardized val
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for data in va_loader:
                    data = data.to(DEVICE, non_blocking=True)
                    preds.append(model(data).cpu().numpy())
                    trues.append(data.y.cpu().numpy())
            preds = np.concatenate(preds); trues = np.concatenate(trues)
            current_mse = np.mean((preds - trues) ** 2)
            if best_fold_mse - current_mse > min_delta:
                best_fold_mse = current_mse
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        best_epochs.append(best_epoch)
        mses.append(best_fold_mse)

    return {"objective": -np.mean(mses), "metadata": {"best_epochs": 1+int(np.median(best_epochs))}}

def inner_objective_ml(hp_dict, model_name, X, y, inner_folds, seed):
    """
    Objective for ML models: average neg-MSE across inner folds.
    Targets are standardized per inner-train fold (StandardScaler on y_train).
    """
    kb = KBinsDiscretizer(n_bins=inner_folds, encode='ordinal', strategy='quantile')
    y_bins = kb.fit_transform(y.reshape(-1,1)).flatten()
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    mses = []
    for train_idx, val_idx in skf.split(X, y_bins):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # --- standardize y on inner-train only ---
        yss = _fit_array_y_scaler(y_tr)
        y_tr_s = _transform_like(yss, y_tr)
        y_val_s = _transform_like(yss, y_val)

        if model_name == 'MLP':
            loader_tr = create_ml_loader(X_tr, y_tr_s, batch_size=hp_dict['batch_size'])
            loader_val = create_ml_loader(X_val, y_val_s, batch_size=hp_dict['batch_size'])
            model = MLP(X.shape[1], hp_dict).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
            criterion = torch.nn.MSELoss()
            # Early‐stopping settings
            patience = hp_dict.get('patience', 5)
            min_delta = 1e-4
            best_val_mse = float('inf')
            epochs_no_improve = 0
            for _ in range(hp_dict['epochs']):
                model.train()
                for xb, yb in loader_tr:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for xb, yb in loader_val:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        out = model(xb)
                        preds.append(out.cpu().numpy())
                        trues.append(yb.cpu().numpy())
                preds = np.concatenate(preds)
                trues = np.concatenate(trues)
                val_mse = np.mean((preds - trues) ** 2)

                if best_val_mse - val_mse > min_delta:
                    best_val_mse = val_mse
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            mses.append(best_val_mse)
        elif model_name == 'XGB':
            cls = ML_MODELS[model_name]
            model = cls(hp_dict)
            # train on standardized targets
            model.fit(X_tr.numpy(), y_tr_s.numpy() if hasattr(y_tr_s, "numpy") else y_tr_s, verbose=False)
            preds_s = model.predict(X_val.numpy())
            # compare in standardized space
            yv_s = y_val_s if isinstance(y_val_s, np.ndarray) else y_val_s.numpy()
            mses.append(np.mean((preds_s - yv_s)**2))
        else:
            cls = ML_MODELS[model_name]
            model = cls(hp_dict)
            model.fit(X_tr.numpy(), y_tr_s.numpy() if hasattr(y_tr_s, "numpy") else y_tr_s)
            preds_s = model.predict(X_val.numpy())
            yv_s = y_val_s if isinstance(y_val_s, np.ndarray) else y_val_s.numpy()
            mses.append(np.mean((preds_s - yv_s)**2))
    return -np.mean(mses)


def nested_cv_for_model(model_name, train_graphs, features, labels, args):
    outer_results = []
    # Stratify outer folds
    if model_name in GRAPH_MODELS:
        ys = np.array([g.y.item() for g in train_graphs])
        kb = KBinsDiscretizer(n_bins=args.outer_folds, encode='ordinal', strategy='quantile')
        y_bins = kb.fit_transform(ys.reshape(-1,1)).flatten()
    else:
        y_bins = KBinsDiscretizer(n_bins=args.outer_folds, encode='ordinal', strategy='quantile').fit_transform(labels.reshape(-1,1)).flatten()

    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed + rep)
        for fold, (train_idx, test_idx) in enumerate(skf.split(train_graphs if model_name in GRAPH_MODELS else features, y_bins)):
            # Model path
            log_dir = os.path.join(args.model_dir, 'hpo', model_name, f'rep{rep}_fold{fold}')
            os.makedirs(log_dir, exist_ok=True)

            # Prepare data for this outer split
            if model_name in GRAPH_MODELS:
                tr_graphs_raw = [train_graphs[i] for i in train_idx]
                te_graphs_raw = [train_graphs[i] for i in test_idx]
            else:
                X_tr, X_te = features[train_idx], features[test_idx]
                y_tr, y_te = labels[train_idx], labels[test_idx]

            # Define HpProblem
            if model_name in GRAPH_MODELS:
                hp_problem = define_hp_problem_graph(model_name)
            else:
                hp_problem = define_hp_problem_ml(model_name)

            # Create evaluator (inner objective already standardizes y per inner-train fold)
            def run_inner(hp_config):
                if model_name in GRAPH_MODELS:
                    return inner_objective_graph(hp_config, model_name, tr_graphs_raw, args.inner_folds, args.seed+rep)
                else:
                    return inner_objective_ml(hp_config, model_name, features[train_idx], labels[train_idx], args.inner_folds, args.seed+rep)

            evaluator = Evaluator.create(
                run_function=run_inner,
                method="ray",
                method_kwargs={
                        "num_cpus": args.num_cpus,
                        "num_gpus": args.num_gpus,
                        "num_cpus_per_task": 1,
                        "num_gpus_per_task": 1,
                },
            )
            search = CBO(hp_problem, evaluator,
                         initial_points=[hp_problem.default_configuration],
                         acq_optimizer='mixedga',
                         log_dir=log_dir)
            history = search.search(max_evals=args.inner_evals)
            best_idx = history['objective'].idxmax()
            best_row = history.loc[best_idx]

            # Build best_hps with proper types
            best_hps = {}
            for col in history.columns:
                if col.startswith('p:'):
                    key = col.split('p:')[1]
                    raw_val = best_row[col]
                    if isinstance(raw_val, str):
                        try:
                            val = ast.literal_eval(raw_val)
                        except (ValueError, SyntaxError):
                            val = raw_val
                    else:
                        val = raw_val
                    if isinstance(val, np.generic):
                        val = val.item()
                    best_hps[key] = val
                elif col.startswith("m:best"):
                    best_hps[col.split("m:", 1)[1]] = best_row[col].item()
            
            # --------------------- Retrain on outer train (standardized y) ---------------------
            start = time.time()
            if model_name in GRAPH_MODELS:
                # fit scaler on outer-train y and standardize tr/te graphs' y
                y_mean, y_std = _fit_graph_y_scaler(tr_graphs_raw)
                tr_graphs = _apply_graph_y_standardize(tr_graphs_raw, y_mean, y_std)
                te_graphs = _apply_graph_y_standardize(te_graphs_raw, y_mean, y_std)

                loader_tr = create_graph_loader(tr_graphs, batch_size=best_hps.get('batch_size',32), shuffle=True, num_workers=8, pin_memory=True)
                loader_te = create_graph_loader(te_graphs, batch_size=best_hps.get('batch_size',32), shuffle=False, num_workers=8, pin_memory=True)
                model = GRAPH_MODELS[model_name](train_graphs[0].x.size(1), best_hps).to(DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=best_hps.get('lr',1e-3))
                criterion = torch.nn.MSELoss()
                num_epochs = max(best_hps.get('best_epochs',50), 50)
                for _ in range(num_epochs):
                    model.train()
                    for data in loader_tr:
                        data=data.to(DEVICE) 
                        optimizer.zero_grad() 
                        loss=criterion(model(data), data.y.float())   # y is standardized
                        loss.backward() 
                        optimizer.step()
                model.eval() 
                preds_s, trues_s = [], []
                with torch.no_grad():
                    for data in loader_te:
                        data=data.to(DEVICE) 
                        preds_s.append(model(data).cpu().numpy()) 
                        trues_s.append(data.y.cpu().numpy())       # standardized y
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
                preds_s, trues_s = np.concatenate(preds_s), np.concatenate(trues_s)
                # inverse-transform to original units for metrics/plots
                preds = _inverse_standardize(preds_s, y_mean, y_std)
                trues = _inverse_standardize(trues_s, y_mean, y_std)
            else:
                # ML case: fit StandardScaler on outer-train y, train on standardized, invert at eval
                yss = _fit_array_y_scaler(y_tr)
                y_tr_s = _transform_like(yss, y_tr)
                y_te_s = _transform_like(yss, y_te)

                if model_name == 'MLP':
                    loader_tr = create_ml_loader(X_tr, y_tr_s, batch_size=best_hps['batch_size'], shuffle=True)
                    loader_te = create_ml_loader(X_te, y_te_s, batch_size=best_hps['batch_size'], shuffle=False)
                    model = MLP(X_tr.shape[1],best_hps).to(DEVICE)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=best_hps['lr'])
                    criterion = torch.nn.MSELoss()
                    for _ in range(best_hps.get('epochs',100)):
                        model.train()
                        for xb, yb in loader_tr:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            optimizer.zero_grad()
                            loss = criterion(model(xb), yb)  # standardized target
                            loss.backward()
                            optimizer.step()
                    model.eval()
                    preds_s, trues_s = [], []
                    with torch.no_grad():
                        for xb, yb in loader_te:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            out = model(xb)
                            preds_s.append(out.cpu().numpy())
                            trues_s.append(yb.cpu().numpy())
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
                    preds_s, trues_s = np.concatenate(preds_s), np.concatenate(trues_s)
                    # inverse-transform to original units
                    preds = _inverse_transform_like(yss, preds_s)
                    trues = _inverse_transform_like(yss, trues_s)
                elif model_name == 'XGB':
                    best_hps['early_stopping_rounds'] = None
                    cls = ML_MODELS[model_name]
                    model = cls(best_hps)
                    model.fit(X_tr.numpy(), y_tr_s.numpy() if hasattr(y_tr_s, "numpy") else y_tr_s, verbose=False)
                    preds_s = model.predict(X_te.numpy())
                    # inverse-transform prediction to original units; compare to original y_te
                    preds = _inverse_transform_like(yss, preds_s)
                    trues = y_te  # already original units
                    pickle.dump(model, open(os.path.join(log_dir, 'best_model.pkl'),'wb'))
                else:
                    model = ML_MODELS[model_name](best_hps)
                    model.fit(X_tr.numpy(), y_tr_s.numpy() if hasattr(y_tr_s, "numpy") else y_tr_s)
                    preds_s = model.predict(X_te.numpy())
                    preds = _inverse_transform_like(yss, preds_s)
                    trues = y_te
                    pickle.dump(model, open(os.path.join(log_dir, 'best_model.pkl'),'wb'))

            elapsed = time.time() - start
            outer_mse = np.mean((preds - trues)**2)

            # Prepare per-fold output dir
            subdir = os.path.join(args.output_dir, model_name, f"rep{rep}_fold{fold}")
            os.makedirs(subdir, exist_ok=True)

            # Evaluate and produce parity plot + CI band (in original units)
            metrics = evaluate_and_plot(
                y_true=trues,
                y_pred=preds,
                plot_path=os.path.join(subdir, "parity.png"),
                metrics_path=os.path.join(subdir, "metrics.json")
            )
            metrics['mse'] = outer_mse

            outer_results.append({
                'rep': rep,
                'fold': fold,
                'train_idx': train_idx.tolist(),
                'test_idx':  test_idx.tolist(),
                'time': elapsed,
                'hps': best_hps,
                'metrics': metrics
            })
            
            # del model, optimizer, loader_tr, loader_te
            # torch.cuda.empty_cache()
            
    return outer_results


def main():
    parser = argparse.ArgumentParser(description='Nested nested cross-validation')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--outer_folds', type=int, default=5)
    parser.add_argument('--inner_folds', type=int, default=5) 
    parser.add_argument('--inner_evals', type=int, default=100) 
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load combined train+val
    train_graphs = load_pyg_data(args.data_dir)
    features, labels = get_ml_features(train_graphs)

    all_results = {}
    for model_name in args.models:
        print(f"Running nested CV for {model_name}...")
        res = nested_cv_for_model(model_name, train_graphs, features, labels, args)
        all_results[model_name] = res
        # save per-model results
        with open(os.path.join(args.output_dir, f"{model_name}_nested_results.json"), 'w') as f:
            json.dump(res, f, indent=2)
    print("Nested CV complete. Results saved to", args.output_dir)

if __name__ == '__main__':
    main()