from __future__ import annotations

import os
import re
import sys
import pdb
import json
import pickle
import argparse
from typing import List, Tuple, Dict, Set

import fnmatch
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import tarfile  # <-- added

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils import get_node_features, compute_rac
from xyz2graph import MolGraph

def read_codes(codes_path: str) -> Set[str]:
    # use the provided path (so it reads from tmqm_dataset/)
    with open(codes_path, "rb") as f:
        codes = pickle.load(f)
    return codes

def filename_from_comment(comment: str, fallback="molecule"):
    m = re.search(r"CSD_code\s*=\s*([A-Za-z0-9]+)", comment)
    return f"{m.group(1)}.xyz" if m else f"{fallback}.xyz"

def write_xyz_block(block: list[str], path: str | None = None, strict: bool = True) -> str:
    """
    block: ['N_atoms', 'comment', 'elem x y z', ...]
    path:  optional output path; if None, derive name from comment
    strict: if True, raise if len(atoms) != N; if False, truncate/allow.
    Returns the written file path.
    """
    if len(block) < 2:
        raise ValueError("Block must have at least atom count and comment.")

    try:
        n = int(block[0].strip())
    except ValueError as e:
        raise ValueError("First item must be an integer atom count.") from e

    comment = block[1].rstrip("\n")
    atoms = [line.rstrip("\n") for line in block[2:]]

    if strict and len(atoms) != n:
        raise ValueError(f"Expected {n} atom lines, got {len(atoms)}.")
    elif not strict and len(atoms) < n:
        raise ValueError(f"Only {len(atoms)} atom lines provided; need {n}.")
    else:
        atoms = atoms[:n]  # in case extra lines are present when strict=False

    if path is None:
        path = os.path.join('selected_xyz', filename_from_comment(comment))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n{comment}\n")
        f.write("\n".join(atoms))
        f.write("\n")  # final newline for POSIX friendliness
    return path

def xyz_to_pyg(xyz_path: str, code: str | None = None) -> Data:
    # --- graph build (same as before) ---
    mg = MolGraph()
    mg.read_xyz(xyz_path)
    G = mg.to_networkx()

    # --- nodes → arrays (no sorting costs; keep native order but remap IDs) ---
    node_ids, node_dicts = zip(*G.nodes(data=True)) if G.number_of_nodes() else ([], [])
    n = len(node_ids)
    id2idx = {nid: i for i, nid in enumerate(node_ids)}

    elements = [d["element"] for d in node_dicts]
    coords   = np.asarray([d["xyz"]     for d in node_dicts], dtype=np.float32)  # (N,3)
    pos      = torch.from_numpy(coords)

    # Node features (vectorized stack, float32)
    Zn, Rcn, wn, chin = get_node_features(elements)  # expected array-like
    Zn   = np.asarray(Zn,   dtype=np.float32)
    Rcn  = np.asarray(Rcn,  dtype=np.float32)
    wn   = np.asarray(wn,   dtype=np.float32)
    chin = np.asarray(chin, dtype=np.float32)
    x = torch.from_numpy(np.column_stack((Zn, Rcn, wn, chin)))  # (N,4)

    # --- edges → arrays (vectorized) ---
    if G.number_of_edges() == 0:
        # Handle edge-less molecules gracefully
        edge_index   = torch.empty((2, 0), dtype=torch.long)
        length       = torch.empty((0, 1), dtype=torch.float32)
        edge_feature = torch.empty((0, 4), dtype=torch.float32)
    else:
        # Get edge endpoints and remap to [0..N-1]
        E = G.number_of_edges()
        u_iter = (id2idx[u] for u, v in G.edges())
        v_iter = (id2idx[v] for u, v in G.edges())
        row = np.fromiter(u_iter, count=E, dtype=np.int64)
        col = np.fromiter(v_iter, count=E, dtype=np.int64)
        edge_index = torch.from_numpy(np.vstack((row, col)))

        # Distances (vectorized)
        diff = coords[row] - coords[col]                 # (E,3)
        length_np = np.linalg.norm(diff, axis=1)         # (E,)
        length = torch.from_numpy(length_np[:, None])    # (E,1), float32

        # Chemistry-inspired edge features (vectorized gathers)
        chi_diff = (chin[row] - chin[col])[:, None]
        Z_prod   = (Zn[row]   * Zn[col])[:, None]
        Rc_diff  = (Rcn[row]  - Rcn[col])[:, None]
        Rc_prod  = (Rcn[row]  * Rcn[col])[:, None]
        edge_feature = torch.from_numpy(
            np.concatenate((chi_diff, Z_prod, Rc_diff, Rc_prod), axis=1).astype(np.float32)
        )

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=length,
        edge_feature=edge_feature,
    )

    if code is None:
        code = os.path.splitext(os.path.basename(xyz_path))[0]
    data.code = code
    data.y = None
    return data

def build_pyg_dataset_from_dir(xyz_dir: str, save_path: str):
    xyz_files = sorted(
        [os.path.join(xyz_dir, f) for f in os.listdir(xyz_dir) if f.lower().endswith(".xyz")]
    )
    data_list = []
    for p in tqdm(xyz_files, desc="Building PyG objects"):
        code = os.path.splitext(os.path.basename(p))[0]
        data = xyz_to_pyg(p, code=code)
        data_list.append(data)
    torch.save(data_list, save_path)
    return data_list

# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CSD codes from xyzs and build PyG objects.")
    # g = p.add_mutually_exclusive_group(required=True)
    p.add_argument("--xyz_dir", default="tmqm_dataset", help="Directory containing XYZ files")
    p.add_argument("--outdir", default="selected_xyz", help="Where to write the extracted per-molecule .xyz files")
    p.add_argument("--save", default="data.pt", help="Output path for torch-saved list[Data]")
    return p.parse_args()

def main():
    # >>> added: resolve sibling tmqm_dataset folder (two levels up from this file) <<<
    project_root = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
    tmqm_dir = os.path.join(project_root, "tmqm_dataset")
    os.makedirs(tmqm_dir, exist_ok=True)

    # >>> added: extract tmqm_dataset.tar.gz if needed into tmqm_dir <<<
    tar_path = os.path.join(tmqm_dir, "tmqm_dataset.tar.gz")
    if os.path.exists(tar_path):
        all_xyz_dir = os.path.join(tmqm_dir, "all_xyz")
        codes_pkl   = os.path.join(tmqm_dir, "tmqm_names.pkl")
        if not (os.path.isdir(all_xyz_dir) and os.path.isfile(codes_pkl)):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=tmqm_dir)

    args = parse_args()

    # >>> added: force everything to live in tmqm_dir <<<
    args.xyz_dir = os.path.join(tmqm_dir, "all_xyz")
    args.codes   = os.path.join(tmqm_dir, "tmqm_names.pkl")
    args.outdir  = os.path.join(tmqm_dir, "selected_xyz")
    args.save    = os.path.join(tmqm_dir, "tmqm_data.pt")

    os.makedirs(args.outdir, exist_ok=True)

    codes = read_codes(args.codes)
    xyz = []
    for i in range(3):
        with open(os.path.join(args.xyz_dir, f'tmQM_X{i+1}.xyz')) as file:
            for line in file:
                xyz.append(line.rstrip())

    size = len(xyz)
    counter = 1
    while counter < size:
        n_lines = int(xyz[counter-1])
        code = xyz[counter].split('|')[0].split('=')[1].strip()
        if code in codes:
            _ = write_xyz_block(
                xyz[counter-1:counter+n_lines+1],
                path=os.path.join(args.outdir, f"{code}.xyz")  # ensure it lands in tmqm_dataset/selected_xyz
            )
        counter += (n_lines + 3)

    build_pyg_dataset_from_dir(args.outdir, args.save)  # -> tmqm_dataset/tmqm_data.pt
    print("Done.")

if __name__ == "__main__":
    main()
