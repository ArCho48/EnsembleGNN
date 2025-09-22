import os
import pdb
import json
import argparse
import numpy as np
from collections import defaultdict
import torch
torch.backends.cudnn.benchmark = True
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import warnings
from tqdm.auto import tqdm  
try:
    from joblib import dump as joblib_dump
    _HAS_JOBLIB = True
except Exception:
    import pickle
    _HAS_JOBLIB = False

from dataloader import ( load_combined_pickle_data, create_graph_loader, 
                        create_ml_loader, load_pyg_data, get_ml_features )
from models import (
    GCNNet, SAGENet, GATNet, GINNet, GINENet,
    PNANet, TransformerNet, DimeNetNet, SchNetNet,
    MLP, get_rf_model, get_svm_model, get_xgb_model
)
from hp_problem import define_hp_problem

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

# -----------------------------
# Helpers
# -----------------------------
def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _as_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)

def _stratify_from_labels(labels, n_bins=10):
    """Return discrete labels for stratification. If labels look continuous,
    bin them into quantiles. If stratification is impossible, return None."""
    y = np.asarray(labels).ravel()
    uniq = np.unique(y)
    if (np.issubdtype(y.dtype, np.integer) and len(uniq) >= 2) or (len(uniq) <= n_bins and len(uniq) >= 2):
        return y
    try:
        qs = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(y, qs))
        if len(edges) <= 2:
            return None
        bins = edges[1:-1]
        strat = np.digitize(y, bins, right=False)
        if len(np.unique(strat)) < 2:
            return None
        return strat
    except Exception:
        return None

def _label_from_graph(g):
    """Extract a scalar class label from g.y (handles tensor, one-hot)."""
    y = g.y
    if torch.is_tensor(y):
        y = y.detach().cpu()
        if y.ndim == 0 or (y.ndim == 1 and y.numel() == 1):
            return int(y.item())
        return int(torch.argmax(y).item())
    arr = np.asarray(y)
    if arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1):
        return int(arr.item())
    return int(arr.argmax())

def stratified_split_graph(graphs, test_size=0.1, val_size=0.1, random_state=42):
    """
    True 3-way stratification by allocating per class directly into
    test/val/train according to proportions. Works even when some classes
    are tiny (they’ll be assigned to fewer than 3 splits if needed).
    """
    assert 0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1
    test_p, val_p, train_p = test_size, val_size, 1.0 - test_size - val_size
    props = np.array([test_p, val_p, train_p])

    by_class = defaultdict(list)
    for i, g in enumerate(graphs):
        by_class[_label_from_graph(g)].append(i)

    rng = np.random.default_rng(random_state)
    test_idx, val_idx, train_idx = [], [], []

    for _, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)

        # ideal fractional allocation
        alloc = props * n
        base = np.floor(alloc).astype(int)
        rem = n - base.sum()
        if rem > 0:
            # give the remaining samples to splits with largest fractional parts
            frac = alloc - base
            for j in np.argsort(frac)[::-1][:rem]:
                base[j] += 1

        n_test, n_val, n_train = base.tolist()
        # slice
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test + n_val])
        train_idx.extend(idxs[n_test + n_val:])

    # Map indices back to graphs
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    
    # --------- NEW: standardize targets using TRAIN stats ---------
    # Stack train targets as (N, C). Works for scalar or vector y.
    ys_train = torch.cat([g.y.float().view(1, -1) for g in train_graphs], dim=0)
    y_mean = ys_train.mean(dim=0)                             # (C,)
    y_std  = ys_train.std(dim=0, unbiased=False).clamp_min(1e-8)  # avoid div-by-zero

    def _apply_standardize(gs):
        for g in gs:
            y = g.y.float().view(1, -1)
            y_norm = (y - y_mean) / y_std
            g.y = y_norm.view_as(g.y)
        return gs

    _apply_standardize(train_graphs)
    _apply_standardize(val_graphs)
    _apply_standardize(test_graphs)
    # -------------------------------------------------------------

    return train_graphs, val_graphs, test_graphs

def train_graph_model(input_dim, hp_dict, model_name, train_loader, val_loader, test_loader, model_dir):
    model = GRAPH_MODELS[model_name](input_dim, hp_dict, edge_dim=5).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
    criterion = torch.nn.MSELoss()

    patience = hp_dict.get('patience', 5)
    min_delta = 1e-4
    best_mse = float('inf')
    best_path = os.path.join(model_dir, f"{model_name}_best.pt")
    epochs_no_improve = 0
    num_epochs = int(hp_dict['epochs'])

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        se_sum, n_elem = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} • train", leave=False)
        for data in train_bar:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()

            # track train running MSE
            with torch.no_grad():
                se_sum += torch.sum((out - data.y.float()) ** 2).item()
                n_elem += data.y.numel()
                train_bar.set_postfix(train_mse=se_sum / max(1, n_elem))

        # ---- Validate ---- (streaming MSE so tqdm can show live metric)
        model.eval()
        se_sum, n_elem = 0.0, 0
        val_bar = tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} • val", leave=False)
        with torch.no_grad():
            for data in val_bar:
                data = data.to(DEVICE)
                out = model(data)
                se_sum += torch.sum((out - data.y.float()) ** 2).item()
                n_elem += data.y.numel()
                val_bar.set_postfix(val_mse=se_sum / max(1, n_elem))
        current_mse = se_sum / max(1, n_elem)

        # ---- Early stopping check ----
        if best_mse - current_mse > min_delta or best_mse == float('inf'):
            best_mse = current_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"[{model_name}] e{epoch+1} • best val MSE: {best_mse:.6f}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            tqdm.write(f"[{model_name}] Early stopping at epoch {epoch+1} (best val MSE {best_mse:.6f})")
            break

    # ---- Final eval on TEST with best checkpoint ----
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    se_sum = 0.0
    n_elem = 0
    test_bar = tqdm(test_loader, desc=f"[{model_name}] test", leave=False)
    with torch.no_grad():
        for data in test_bar:
            data = data.to(DEVICE)
            out = model(data).float()
            y = data.y.float()
            se_sum += torch.sum((out - y) ** 2).item()
            n_elem += y.numel()
            test_bar.set_postfix(test_mse=se_sum / max(1, n_elem))
    test_mse = se_sum / max(1, n_elem)

    metrics = {
        'val_mse': best_mse,
        'test_mse': test_mse,
    }
    return metrics, best_path

def train_ml_model(input_dim, hp_dict, model_name, features, labels, model_dir):
    test_size = float(hp_dict.get('test_size', 0.1))
    val_size  = float(hp_dict.get('val_size', 0.1))
    assert 0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1, \
        "test_size and val_size must be in (0,1) and sum to < 1."

    X = _as_numpy(features)
    y = _as_numpy(labels).reshape(-1, 1)
    y_flat = y.ravel()

    y_strat = _stratify_from_labels(y_flat, n_bins=int(hp_dict.get('strat_bins', 10)))

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=int(hp_dict.get('random_state', 42)),
        stratify=y_strat if y_strat is not None else None
    )
    rel_test = test_size / (test_size + val_size)
    y_tmp_strat = _stratify_from_labels(y_tmp.ravel(), n_bins=int(hp_dict.get('strat_bins', 10)))

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=int(hp_dict.get('random_state', 42)),
        stratify=y_tmp_strat if y_tmp_strat is not None else None
    )

    if model_name == 'MLP':
        batch_size = int(hp_dict.get('batch_size', 32))
        train_loader = create_ml_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader   = create_ml_loader(X_val,   y_val,   batch_size=batch_size, shuffle=False)
        test_loader  = create_ml_loader(X_test,  y_test,  batch_size=batch_size, shuffle=False)

        model = MLP(input_dim, hp_dict).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp_dict['lr'])
        criterion = torch.nn.MSELoss()

        patience = hp_dict.get('patience', 5)
        min_delta = 1e-4
        best_val_mse = float('inf')
        best_path = os.path.join(model_dir, f"{model_name}_best.pt")
        epochs_no_improve = 0
        num_epochs = int(hp_dict['epochs'])

        for epoch in range(num_epochs):
            # --- Train ---
            model.train()
            se_sum, n_elem = 0.0, 0
            train_bar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} • train", leave=False)
            for xb, yb in train_bar:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    se_sum += torch.sum((out - yb) ** 2).item()
                    n_elem += yb.numel()
                    train_bar.set_postfix(train_mse=se_sum / max(1, n_elem))

            # --- Validate ---
            model.eval()
            se_sum, n_elem = 0.0, 0
            val_bar = tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} • val", leave=False)
            with torch.no_grad():
                for xb, yb in val_bar:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    out = model(xb)
                    se_sum += torch.sum((out - yb) ** 2).item()
                    n_elem += yb.numel()
                    val_bar.set_postfix(val_mse=se_sum / max(1, n_elem))
            val_mse = se_sum / max(1, n_elem)

            if best_val_mse - val_mse > min_delta or best_val_mse == float('inf'):
                best_val_mse = val_mse
                torch.save(model.state_dict(), best_path)
                epochs_no_improve = 0
                tqdm.write(f"[{model_name}] e{epoch+1} • best val MSE: {best_val_mse:.6f}")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                tqdm.write(f"[{model_name}] Early stopping at epoch {epoch+1} (best val MSE {best_val_mse:.6f})")
                break

        # --- Test with best checkpoint ---
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        se_sum, n_elem = 0.0, 0
        test_bar = tqdm(test_loader, desc=f"[{model_name}] test", leave=False)
        with torch.no_grad():
            for xb, yb in test_bar:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                se_sum += torch.sum((out - yb) ** 2).item()
                n_elem += yb.numel()
                test_bar.set_postfix(test_mse=se_sum / max(1, n_elem))
        test_mse = se_sum / max(1, n_elem)

        metrics = {'val_mse': best_val_mse, 'test_mse': test_mse}
        path = best_path

    elif model_name == 'XGB':
        cls = ML_MODELS[model_name]
        model = cls(hp_dict)

        fit_desc = f"[{model_name}] fit (train+val)"
        with tqdm(total=1, desc=fit_desc, leave=False) as pbar:
            model.fit(X_train, y_train.ravel(),
                      eval_set=[(X_val, y_val.ravel())],
                      verbose=False)
            pbar.update(1)

        val_preds  = model.predict(X_val)
        test_preds = model.predict(X_test)
        val_mse  = mean_squared_error(y_val.ravel(),  val_preds)
        test_mse = mean_squared_error(y_test.ravel(), test_preds)

        best_path = os.path.join(model_dir, f"{model_name}_best.joblib")
        if _HAS_JOBLIB:
            joblib_dump(model, best_path)
        else:
            with open(best_path, 'wb') as f:
                pickle.dump(model, f)

        metrics = {'val_mse': val_mse, 'test_mse': test_mse}
        path = best_path

    else:
        # Generic sklearn-style model
        cls = ML_MODELS[model_name]
        model = cls(hp_dict)

        fit_desc = f"[{model_name}] fit (train)"
        with tqdm(total=1, desc=fit_desc, leave=False) as pbar:
            model.fit(X_train, y_train.ravel())
            pbar.update(1)

        val_preds  = model.predict(X_val)
        test_preds = model.predict(X_test)
        val_mse  = mean_squared_error(y_val.ravel(),  val_preds)
        test_mse = mean_squared_error(y_test.ravel(), test_preds)

        best_path = os.path.join(model_dir, f"{model_name}_best.joblib")
        if _HAS_JOBLIB:
            joblib_dump(model, best_path)
        else:
            with open(best_path, 'wb') as f:
                pickle.dump(model, f)

        metrics = {'val_mse': val_mse, 'test_mse': test_mse}
        path = best_path

    return metrics, path

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model names to train')
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # load HP config
    hp = define_hp_problem(args.models[0])
    hp_config = hp.default_configuration

    # Load combined train+val
    all_graphs = load_pyg_data(args.data_dir)
    all_features, all_labels = get_ml_features(all_graphs)

    # create loaders
    graph_train, graph_val, graph_test = stratified_split_graph(all_graphs, test_size=0.2, val_size=0.1) 
    train_loader = create_graph_loader(graph_train, batch_size=hp_config.get('batch_size',32))
    val_loader = create_graph_loader(graph_val, batch_size=hp_config.get('batch_size',32), shuffle=False)
    test_loader = create_graph_loader(graph_test, batch_size=hp_config.get('batch_size',32), shuffle=False)

    results = {}
    for name in tqdm(args.models, desc="Training models"):
        if name in GRAPH_MODELS:
            metrics, path = train_graph_model(
                all_graphs[0].x.shape[1], 
                hp_config,
                name,
                train_loader,
                val_loader,
                test_loader,
                args.model_dir,
            )
        elif name in ML_MODELS:
            metrics, path = train_ml_model(
                all_features.shape[1],
                hp_config,
                name,  
                all_features,
                all_labels,
                args.model_dir,
            )
        else:
            raise ValueError(f"Unknown model: {name}")
        results[name] = {'metrics': metrics, 'path': path}
        tqdm.write(f"[{name}] done: {metrics}")

    # Convert any numpy/torch types to plain Python floats/ints
    results[name]['metrics']['val_mse'] = float(results[name]['metrics']['val_mse'])
    results[name]['metrics']['test_mse'] = float(results[name]['metrics']['test_mse'])

    # save overall results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
