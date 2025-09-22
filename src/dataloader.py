
import os
import pickle
import pdb
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_pyg_data(path: str) -> list[Data]:
    if os.path.isfile(path) and path.endswith('.pt'):
        loaded = torch.load(path, weights_only=False)
        # Expecting either a single Data or a list of Data
        if isinstance(loaded, Data):
            return [loaded]
        elif isinstance(loaded, (list, tuple)):
            return list(loaded)
        else:
            raise ValueError(f"Unsupported .pt content: {type(loaded)}")
    elif os.path.isdir(path):
        pattern = os.path.join(path, '*.pt')
        file_paths = sorted(glob.glob(pattern))
        data_list = []
        for fp in file_paths:
            data = torch.load(fp, weights_only=False)
            if isinstance(data, Data):
                data_list.append(data)
            elif isinstance(data, (list, tuple)):
                data_list.extend(list(data))
            else:
                raise ValueError(f"Unsupported .pt content in {fp}: {type(data)}")
        return data_list
    else:
        raise FileNotFoundError(f"No file or directory found at: {path}")

def load_pickle_data(path_or_paths) -> list[Data]:
    """
    Load PyTorch Geometric Data objects from one or more pickle files (.pkl).

    Args:
        path_or_paths (str or list of str): Path(s) to .pkl file(s) or directory containing them.

    Returns:
        List[Data]
    """
    # Determine list of pickle file paths
    if isinstance(path_or_paths, str):
        if os.path.isdir(path_or_paths):
            pattern = os.path.join(path_or_paths, '*.pkl')
            file_paths = sorted(glob.glob(pattern))
        elif os.path.isfile(path_or_paths) and path_or_paths.endswith('.pkl'):
            file_paths = [path_or_paths]
        else:
            raise FileNotFoundError(f"No .pkl file or directory found at: {path_or_paths}")
    elif isinstance(path_or_paths, (list, tuple)):
        file_paths = [p for p in path_or_paths if os.path.isfile(p) and p.endswith('.pkl')]
        if not file_paths:
            raise FileNotFoundError(f"No valid .pkl files found in list: {path_or_paths}")
    else:
        raise ValueError("path_or_paths must be str or list of str pointing to .pkl files or directory")

    data_list = []
    for fp in file_paths:
        with open(fp, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, Data):
                data_list.append(loaded)
            elif isinstance(loaded, (list, tuple)):
                data_list.extend(list(loaded))
            else:
                raise ValueError(f"Unsupported pickle content {type(loaded)} in {fp}")
    return data_list


def load_combined_pickle_data(
    datadir: str,
    graphs: bool
) -> tuple[list[Data], torch.Tensor, torch.Tensor]:
    if graphs:
        # Paths
        graph_train_val_pkl = os.path.join(datadir,'train_and_val_graph_list.pkl')
        graph_test_pkl = os.path.join(datadir,'test_graph_list.pkl')

        # Graph splits
        train_val_graphs = load_pickle_data(graph_train_val_pkl)
        test_graphs = load_pickle_data(graph_test_pkl)
        all_graphs = train_val_graphs + test_graphs

        return all_graphs
    else:
        ml_train_val_pkl = os.path.join(datadir,'train_and_val_list.pkl')
        ml_test_pkl = os.path.join(datadir,'test_list.pkl')

        # Helper to load ML features
        def _load_ml_data(pkl_path: str) -> tuple[torch.Tensor, torch.Tensor]:
            with open(pkl_path, 'rb') as f:
                loaded = pickle.load(f)
            # Expect dict format
            if isinstance(loaded, dict) and 'feats' in loaded and 'labels' in loaded:
                # convert sparse feats to dense torch.Tensor
                feats_csr = loaded['feats']
                try:
                    feats_arr = feats_csr.toarray()
                except AttributeError:
                    feats_arr = feats_csr.todense()
                feats = torch.from_numpy(feats_arr).float()
                # labels as numpy array
                labs = torch.from_numpy(loaded['labels']).float()
            # Legacy tuple format
            elif isinstance(loaded, tuple) and len(loaded) == 2 and isinstance(loaded[0], torch.Tensor):
                feats, labs = loaded
            # Legacy list of pairs
            elif isinstance(loaded, (list, tuple)):
                feats_list, labs_list = zip(*loaded)
                feats = torch.stack(list(feats_list))
                labs = torch.stack(list(labs_list))
            else:
                raise ValueError(f"Unsupported ML pickle format: {type(loaded)} in {pkl_path}")
            return feats, labs
        
        # ML splits
        feats_tv, labs_tv = _load_ml_data(ml_train_val_pkl)
        feats_test, labs_test = _load_ml_data(ml_test_pkl)
        all_features = torch.cat([feats_tv, feats_test], dim=0)
        all_labels = torch.cat([labs_tv, labs_test], dim=0)

        return all_features, all_labels

def get_ml_features(data_list: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([data.rac for data in data_list])
    labels = torch.stack([data.y.squeeze() for data in data_list])
    return features, labels

def create_graph_loader(
    data_list: list[Data],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers=2,
    pin_memory=True,
) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def create_ml_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers=8,
    pin_memory=True,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(torch.as_tensor(features), torch.as_tensor(labels).squeeze())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)