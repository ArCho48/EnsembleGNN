import os
import glob
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from xyz2graph import MolGraph
from tqdm import tqdm
from utils import get_node_features, normalize_molecule_name, compute_rac


class MoleculeBondGraphDataset(InMemoryDataset):
    """
    Builds PyG graphs from .xyz files under '<root>/geometries/{csv_name}/',
    attaches node/edge features, and reads ground truths from '<root>/energies/{csv_name}.csv'.

    Modifications:
      - `processed_dir` and `raw_dir` both point to `root`, so no '<root>/raw' is created.
      - `data.pt` is written to '<root>/data.pt' (same folder as 'dataset.tar.gz').
      - Uses PyG's collate to save (data, slices) for InMemoryDataset.
    """

    def __init__(self, root, covalent_scale=1.2, transform=None, pre_transform=None):
        self.covalent_scale = covalent_scale
        super().__init__(root, transform, pre_transform)
        # Load processed data/slices
        self.data, self.slices = torch.load(self.processed_paths[0])

    # Keep both "raw" and "processed" directly in `root`
    @property
    def raw_dir(self):
        return self.root  # prevents creation of "<root>/raw"

    @property
    def processed_dir(self):
        return self.root  # saves "<root>/data.pt"

    # We discover inputs via glob in process(), so skip explicit raw file assertions
    @property
    def raw_file_names(self):
        return []  # bypass PyG's raw existence checks and avoid unintended path assumptions

    @property
    def processed_file_names(self):
        return ['data.pt']  # resolves to <root>/data.pt

    def download(self):
        # Nothing to download; data already present in <root>/energies and <root>/geometries
        pass

    def process(self):
        # ---- 1) Read all energy CSVs ----
        csv_paths = glob.glob(os.path.join(self.root, 'energies', '*.csv'))
        df_dict = {
            os.path.splitext(os.path.basename(p))[0]: pd.read_csv(p)
            for p in csv_paths
        }

        target_cols = [
            'M(II) GS energy, ha',
            'M(II) Delta E[H-L], kcal/mol',
            'M(II) Delta G[H-L] (inc. solvent), kcal/mol',
            'Delta E[III-II], eV',
            'Delta Gsolv[III-II], eV',
            'M(II) GS logP',
            'M(III) logP'
        ]

        # Build list of (dataset_name, csv_row, matched_xyz_path)
        mol_entries = []
        for ds_name, df in df_dict.items():
            geo_folder = os.path.join(self.root, 'geometries', ds_name)
            if not os.path.isdir(geo_folder):
                continue
            for _, row in df.iterrows():
                base_name = row['name']  # CSV 'name' matches normalize_molecule_name
                gs_state = row['M(II) ground spin state (GS)']  # 'HS' or 'LS'

                parts = base_name.split('_', 1)
                mod_name = parts[0] + '_II_' + parts[1] if len(parts) > 1 else base_name + '_II'

                # Find .xyz under the dataset's geometry folder
                all_xyz = glob.glob(os.path.join(geo_folder, f"{mod_name}*.xyz"))
                filtered = [p for p in all_xyz if os.path.basename(p).endswith(f"_{gs_state}.xyz")]
                if not filtered:
                    raise FileNotFoundError(
                        f"No matching M(II) xyz for {base_name} with spin {gs_state} in {geo_folder}"
                    )
                # Expect exactly one match
                mol_entries.append((ds_name, row, filtered[0]))

        # ---- 2) Build Data objects ----
        data_list = []
        rac_list = []
        for ds_name, row, xyz_path in tqdm(mol_entries, total=len(mol_entries)):
            # Graph from xyz
            mg = MolGraph()
            mg.read_xyz(xyz_path)
            G = mg.to_networkx()

            # Nodes
            nodes = sorted(G.nodes(data=True), key=lambda x: x[0])
            elements = [d['element'] for _, d in nodes]
            coords = [d['xyz'] for _, d in nodes]
            pos = torch.tensor(coords, dtype=torch.float)

            Zn, Rcn, wn, chin = get_node_features(elements)
            x = torch.tensor(np.vstack([Zn, Rcn, wn, chin]).T, dtype=torch.float)

            # Edges
            edges = list(G.edges(data=True))
            row_idx = [u for u, v, _ in edges]
            col_idx = [v for u, v, _ in edges]
            edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long)

            length = torch.tensor([
                d.get('length', np.linalg.norm(np.array(coords[u]) - np.array(coords[v])))
                for u, v, d in edges
            ], dtype=torch.float).unsqueeze(1)

            chi_diff = torch.tensor([chin[u] - chin[v] for u, v, _ in edges], dtype=torch.float).unsqueeze(1)
            Z_prod = torch.tensor([Zn[u] * Zn[v] for u, v, _ in edges], dtype=torch.float).unsqueeze(1)
            Rc_diff = torch.tensor([Rcn[u] - Rcn[v] for u, v, _ in edges], dtype=torch.float).unsqueeze(1)
            Rc_prod = torch.tensor([Rcn[u] * Rcn[v] for u, v, _ in edges], dtype=torch.float).unsqueeze(1)
            edge_features = torch.cat([chi_diff, Z_prod, Rc_diff, Rc_prod], dim=1)

            # Targets
            gt = torch.tensor(row[target_cols].values.astype(float), dtype=torch.float)
            y = gt[4].unsqueeze(0)  # Delta Gsolv[III-II], eV
            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                edge_attr=length,
                edge_feature=edge_features,
                gt=gt,
                y=y,
            )

            # RAC features
            rac_feats = compute_rac(xyz_path)
            rac_list.append(rac_feats)
            data.rac = torch.tensor(rac_feats, dtype=torch.float)

            data_list.append(data)

        # ---- 3) Optional pre_transform, collate, and save to <root>/data.pt ----
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
