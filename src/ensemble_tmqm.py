#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import pandas as pd
from torch_geometric.loader import DataLoader

torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- import your graph models -----
from models import (
    GCNNet, SAGENet, GATNet, GINNet, GINENet,
    PNANet, TransformerNet, DimeNetNet, SchNetNet,
)
GRAPH_MODELS = {
    'GCN': GCNNet, 'SAGE': SAGENet, 'GAT': GATNet, 'GIN': GINNet, 'GINE': GINENet,
    'PNA': PNANet, 'Transformer': TransformerNet, 'DimeNet': DimeNetNet, 'SchNet': SchNetNet,
}

def _load_history(history_path: str) -> pd.DataFrame:
    return pd.read_csv(history_path)

def _find_trials_root(model_out_root: str) -> str:
    troot = os.path.join(model_out_root, "trials_raw")
    if not os.path.isdir(troot):
        raise FileNotFoundError(f"Trials directory not found: {troot}")
    return troot

def _get_artifact_dir_from_row(row: pd.Series) -> str | None:
    for key in row.index:
        if key.startswith("m:") and "artifact_dir" in key:
            return row[key]
    return None

def _normalize_history_columns(hist: pd.DataFrame) -> pd.DataFrame:
    if "job_id" not in hist.columns:
        jcols = [c for c in hist.columns if c.lower() in ("job_id", "jobid", "job")]
        if jcols:
            hist = hist.rename(columns={jcols[0]: "job_id"})
        else:
            raise KeyError("Could not find a job_id column in history CSV.")
    if "m:val_mse_std" not in hist.columns:
        alt = [c for c in hist.columns if c.startswith("m:") and "val" in c and ("mse" in c or "rmse" in c)]
        if alt:
            hist = hist.rename(columns={alt[0]: "m:val_mse_std"})
        else:
            raise KeyError("Could not find 'm:val_mse_std' (or alt val metric) in history.")
    return hist

def _load_graph_artifacts(artifact_dir: str) -> Dict:
    artifact = Path(artifact_dir)
    hps = json.load(open(artifact / "hps.json"))
    ysc = json.load(open(artifact / "y_scaler.json"))  # {"mean":..., "std":...}
    state_path = artifact / "model.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {artifact_dir}")
    return {"hps": hps, "y_scaler": ysc, "state_dict_path": str(state_path)}

def _choose_bs(hps: Dict, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    return int(hps.get("batch_size_eval", hps.get("batch_size", 32)))

@torch.no_grad()
def _predict_graphs(model_name: str,
                    hps: Dict,
                    state_dict_path: str,
                    graphs: List,
                    y_scaler: Dict,
                    batch_size: Optional[int],
                    num_workers: int = 4) -> np.ndarray:
    """Return predictions in original units for all graphs. Shape [N].
       batch_size=None -> use hps['batch_size_eval' or 'batch_size'].
    """
    bs = _choose_bs(hps, batch_size)
    in_dim = graphs[0].x.size(1)
    net = GRAPH_MODELS[model_name](in_dim, hps).to(DEVICE)
    net.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))
    net.eval()

    # simple OOM-safe fallback: halve bs on OOM
    def _run_with_bs(bs_try: int) -> np.ndarray:
        loader = DataLoader(graphs, batch_size=bs_try, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        preds_std = []
        for data in loader:
            data = data.to(DEVICE, non_blocking=True)
            out = net(data).detach().cpu().numpy()
            preds_std.append(out)
        preds_std = np.concatenate(preds_std, axis=0).reshape(-1)
        return preds_std

    while True:
        try:
            preds_std = _run_with_bs(bs)
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and bs > 1:
                bs = max(1, bs // 2)
                print(f"[WARN] CUDA OOM; retrying with smaller batch_size={bs}")
                torch.cuda.empty_cache()
                continue
            raise

    mean, std = float(y_scaler["mean"]), float(y_scaler["std"])
    preds = preds_std * std + mean
    return preds

def _safe_get_code(g, idx: int) -> str:
    c = getattr(g, "code", None)
    if c is None:
        return f"IDX_{idx:06d}"
    if isinstance(c, (bytes, bytearray)):
        c = c.decode("utf-8", errors="ignore")
    return str(c)

def main():
    ap = argparse.ArgumentParser(description="Top-K ensemble inference on tmQM graphs (no ground truth).")
    ap.add_argument("--data_pt", type=str, required=True, help="Path to torch-saved list")
    ap.add_argument("--output_dir", type=str, required=True, help="HPO root with hpo_single_no_retrain/<model_name>/")
    ap.add_argument("--model_name", type=str, required=True, choices=list(GRAPH_MODELS.keys()))
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=None, help="Override per-trial HPO batch size; default None uses each trial's hps.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--save_csv", type=str, default="tmqm_top100_preds.csv")
    ap.add_argument("--skip_missing", action="store_true", help="Skip missing/broken trials")
    args = ap.parse_args()

    global DEVICE
    DEVICE = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    graphs = torch.load(args.data_pt, map_location="cpu")
    if not isinstance(graphs, list) or len(graphs) == 0:
        raise ValueError("data_pt must contain a non-empty list of PyG Data objects.")
    print(f"[INFO] Loaded {len(graphs)} graphs from {args.data_pt}")

    model_out_root = os.path.join(args.output_dir, "hpo_single_no_retrain", args.model_name)
    history_path = os.path.join(model_out_root, "search_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Cannot find history csv at {history_path}")
    trials_root = _find_trials_root(model_out_root)

    hist = _normalize_history_columns(_load_history(history_path))
    order = hist["m:val_mse_std"].astype(float).sort_values().index.values
    K = min(args.top_k, len(order))
    print(f"[INFO] Using Top-{K} trials by validation metric.")

    preds_list, used_rows = [], []
    for ridx in order[:K]:
        row = hist.loc[ridx]
        art_dir = _get_artifact_dir_from_row(row)
        if not art_dir or not os.path.isdir(art_dir):
            cand = os.path.join(trials_root, os.path.basename(str(art_dir))) if art_dir else None
            if cand and os.path.isdir(cand):
                art_dir = cand
        if not art_dir or not os.path.isdir(art_dir):
            msg = f"[WARN] artifact_dir not found for row {ridx}: {art_dir}"
            if args.skip_missing:
                print(msg + " — skipping.")
                continue
            else:
                raise FileNotFoundError(msg)

        art = _load_graph_artifacts(art_dir)
        bs_used = _choose_bs(art["hps"], args.batch_size)
        print(f"[INFO] Trial row {ridx}: using batch_size={bs_used}")
        try:
            preds = _predict_graphs(args.model_name, art["hps"], art["state_dict_path"],
                                    graphs, art["y_scaler"],
                                    batch_size=args.batch_size, num_workers=args.num_workers)
        except Exception as e:
            msg = f"[WARN] Failed inference for row {ridx} ({art_dir}): {e}"
            if args.skip_missing:
                print(msg + " — skipping.")
                continue
            else:
                raise
        preds_list.append(preds.reshape(1, -1))
        used_rows.append(int(ridx))
        if len(preds_list) % 10 == 0:
            print(f"[INFO] Collected predictions from {len(preds_list)} / {K} members...")

    if len(preds_list) == 0:
        raise RuntimeError("No ensemble members produced predictions. Check artifacts and paths.")
    preds_matrix = np.concatenate(preds_list, axis=0)  # [K_eff, N]
    K_eff, N = preds_matrix.shape
    print(f"[INFO] Final ensemble size: {K_eff} members; {N} molecules.")

    mean_pred = preds_matrix.mean(axis=0)
    std_pred  = preds_matrix.std(axis=0, ddof=1 if K_eff > 1 else 0)

    codes = [_safe_get_code(g, i) for i, g in enumerate(graphs)]
    df = pd.DataFrame({"CSD_code": codes, "mean": mean_pred, "std": std_pred})
    save_path = Path(args.save_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print("[INFO] Saved predictions CSV:", str(save_path))

    indices_csv = save_path.with_name(save_path.stem + "_topk_indices.csv")
    pd.DataFrame({"row_index": used_rows}).to_csv(indices_csv, index=False)
    print("[INFO] Saved ensemble indices CSV:", str(indices_csv))

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
