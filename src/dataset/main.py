from pathlib import Path
import tarfile, pdb
from torch_geometric.loader import DataLoader
from dataset import MoleculeBondGraphDataset


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if not str(target).startswith(str(base)):
            raise RuntimeError("Blocked path traversal attempt in tar file.")
    tar.extractall(path=path)  # On Python 3.12+, prefer: tar.extractall(path=path, filter="data")


def main():
    # This file is in: <project_root>/src/dataset/this_script.py
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # -> <project_root>
    dataset_dir = project_root / "dataset"   # -> <project_root>/dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    archive_path = dataset_dir / "dataset.tar.gz"
    data_root = dataset_dir                    # where energies/ and geometries/ should live

    energies_dir = data_root / "energies"
    geometries_dir = data_root / "geometries"

    if not (energies_dir.exists() and geometries_dir.exists()):
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, data_root)

        # Handle wrapper directory inside the tarball (e.g., dataset/energies, dataset/geometries)
        if not (energies_dir.exists() and geometries_dir.exists()):
            for p in data_root.iterdir():
                if p.is_dir() and (p / "energies").exists() and (p / "geometries").exists():
                    data_root = p
                    energies_dir = data_root / "energies"
                    geometries_dir = data_root / "geometries"
                    break

        if not (energies_dir.exists() and geometries_dir.exists()):
            raise RuntimeError("After extraction, expected 'energies/' and 'geometries/' were not found.")
    pdb.set_trace()
    # Use the extracted folder as the dataset root
    dataset = MoleculeBondGraphDataset(root=str(data_root), covalent_scale=1.2)

    # Example loader usage (uncomment to iterate)
    # loader = DataLoader(dataset, batch_size=16, shuffle=True)
    # for batch in loader:
    #     print(batch)


if __name__ == "__main__":
    main()
