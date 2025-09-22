# Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty

PyTorch implementation of *Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty* ([https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/683e0487c1cb1ecda0ce5640/original/automated-learning-of-gnn-ensembles-for-predicting-redox-potentials-with-uncertainty.pdf](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/683e0487c1cb1ecda0ce5640/original/automated-learning-of-gnn-ensembles-for-predicting-redox-potentials-with-uncertainty.pdf))

## Overview

This library implements the training, selection, and uncertainty quantification of GNN (and non-GNN) models for predicting redox potentials, as presented in [\[1\]](#citation).
It supports three main operations:

1. **Comparison with SOTA** — repeated *nested cross-validation* (inner HPO, outer test) + statistical tests and parity plots.
2. **Ensemble** — single-split *hyperparameter optimization* followed by top-K/greedy ensembling and test-set evaluation.
3. **TMQM predictions & UQ** — *zero-label* inference on tmQM structures with ensemble mean & epistemic uncertainty.

## Dependencies

* **python >= 3.10**
* **pytorch >= 2.5**
* **pytorch-geometric >= 2.6**
* **deephyper**
* **numpy**
* **pandas**
* **scikit-learn**
* **matplotlib**
* **xgboost**
* **scipy**
* **tqdm**
* (Recommended) a Ray backend for DeepHyper

## Structure

* **Data preparation**
    * `dataset.py` — **MoleculeBondGraphDataset** builds PyG graphs from `.xyz` files under `geometries/<name>/`, attaches node/edge features, and pairs them with targets from `energies/<name>.csv`. Writes a single `<root>/data.pt` compatible with `InMemoryDataset`.
    * `main.py` — helper script to extract `dataset/dataset.tar.gz`, verify `energies/` and `geometries/` are present, and instantiate `MoleculeBondGraphDataset` for preprocessing. Includes a safe tar extraction routine.
    * `tmqm_dataset.py` — utilities/CLI to select specific CSD codes from tmQM multi-XYZ files, write per-molecule `.xyz`, and convert them into a saved PyG list (`data.pt`). Also exposes `xyz_to_pyg` and `build_pyg_dataset_from_dir`.
    * `utils.py` — chemistry utilities: normalized atom features (`Zn`, `Rcn`, `wn`, `chin`), RAC-155 feature computation (153 used; metadata removed), and molecule name normalization helpers.


* **Data loading & utilities**

  * `dataloader.py` — load PyG `Data` lists from a `.pt` file or a directory of `*.pt`; optional pickle loaders; mini-batch loaders; optional pre-serialized ML features for non-GNN baselines.
  * `hp_problem.py` — defines search spaces for ML and GNN models (Deephyper `HpProblem`).
  * `models.py` — GNNs (GCN, SAGE, GAT, GIN, GINE, PNA, Transformer, DimeNet, SchNet) and ML baselines (MLP, RF/ET, SVM, XGB).
  * `evaluation.py` — regression metrics + parity plots with a 95% CI band.

* **Comparison with SOTA**

  * `train_nestedCV.py` — repeated nested CV: inner HPO per fold (Deephyper), outer re-train, metrics in original units, parity plots per fold; saves JSON results for each method.
    **CLI:**

    ```
    python train_nestedCV.py \
      --data_dir DATA_PT_OR_DIR \
      --models MODEL ... \
      --outer_folds 5 --inner_folds 5 --inner_evals 100 \
      --repeats 3 --seed 42 --device cpu \
      --model_dir MODELS_DIR --output_dir RESULTS_DIR \
      --num_cpus 4 --num_gpus 4
    ```
  * `summary.py` — aggregates nested-CV runs across methods, runs paired Wilcoxon signed-rank tests (optional Holm correction), and can emit a `summary.json`.
    **CLI:**

    ```
    python summary.py \
      --root_dir RESULTS_DIR \
      --methods METHOD [METHOD ...] \
      --save_json --output_dir OUT_DIR
    ```

* **Ensemble (HPO + ensembling)**

  * `hp_search.py` — single 0.8/0.1/0.1 split, saves **every** trained trial (weights + scalers) and the search ledger (`search_history.csv`, `trials.json`, `best_summary.json`) under `hpo_single_no_retrain/<MODEL>/`.
    **CLI:**

    ```
    python hp_search.py \
      --data_dir DATA_PT_OR_DIR \
      --models MODEL [MODEL ...] \
      --seed 42 --device gpu \
      --model_dir MODELS_DIR --output_dir RESULTS_DIR \
      --max_evals 1200 --num_cpus 4 --num_gpus 4 \
      --strat_bins 10
    ```
  * `ensemble.py` — builds & evaluates ensembles on the *fixed HPO test split*: reports default/best single, Top-K mean, and greedy-forward ensembles; writes summaries & parity plots (with per-point 95% CI based on ensemble variance).
    **CLI:**

    ```
    python ensemble.py \
      --data_dir DATA_PT_OR_DIR \
      --model_name MODEL \
      --output_dir RESULTS_DIR \
      --seed 42 --strat_bins 10 --top_k 100 \
      --device cpu
    ```

* **TMQM predictions and UQ**

  * `ensemble_tmqm.py` — Top-K ensemble *inference* on tmQM PyG graphs (no ground truth needed). Outputs CSV with `mean` and `std` (epistemic).
    **CLI:**

    ```
    python ensemble_tmqm.py \
      --data_pt TMQM_PT_LIST \
      --output_dir RESULTS_DIR \
      --model_name MODEL \
      --top_k 100 --batch_size B --num_workers 4 \
      --device cuda --save_csv tmqm_top100_preds.csv --skip_missing
    ```

* **optional utility**

  * `train_single.py` — lightweight single-run trainer for a quick baseline (graph & ML). Saves best checkpoint and `results.json`.

> **Data expectations.** For GNN ops, pass a `.pt` file containing a Python list of PyG `Data` objects **or** a directory with multiple `*.pt`. Each `Data` should have `.x`, `.edge_index`, and scalar `.y`. If using non-GNN baselines, include pre-serialized features (e.g., `rac`) as needed.

## Usage

Please cite \[[1](#citation)] in your work when using this library in your experiments.

<!-- ### 1) Comparison with SOTA (nested CV → summary & evaluate)

```bash
# Nested CV (inner HPO per fold, outer evaluation)
python train_nestedCV.py \
  --data_dir Data/train_graphs.pt \
  --models SAGE XGB \
  --outer_folds 5 --inner_folds 5 --inner_evals 100 \
  --repeats 3 --seed 42 --device cpu \
  --model_dir models/ --output_dir results/nestedcv/
```

```bash
# Aggregate + pairwise stats (Holm-adjusted p-values) and optional summary.json
python summary.py \
  --root_dir results/nestedcv \
  --methods SAGE XGB \
  --save_json --output_dir results/nestedcv
```

### 2) Ensemble (HPO search → ensemble evaluate)

```bash
# Hyperparameter search (saves every trained trial + artifacts)
python hp_search.py \
  --data_dir Data/train_graphs.pt \
  --models GCN SAGE GAT \
  --max_evals 1200 --seed 42 --device gpu \
  --model_dir models/ --output_dir results/ --strat_bins 10
```

```bash
# Build & score ensembles on the fixed HPO split
python ensemble.py \
  --data_dir Data/train_graphs.pt \
  --model_name SAGE \
  --output_dir results \
  --top_k 100 --device cpu
```

### 3) TMQM predictions and UQ (no labels)

```bash
# Top-K ensemble inference on tmQM graphs (list saved via torch.save)
python ensemble_tmqm.py \
  --data_pt Data/tmqm_graphs.pt \
  --output_dir results \
  --model_name SAGE \
  --top_k 100 --device cuda \
  --save_csv tmqm_top100_preds.csv --skip_missing
``` -->

## Feedback

For questions and comments, feel free to contact [Arindam Chowdhury](mailto:chowdhurya1@ornl.gov).

## Citation

```
[1] Chowdhury A, Harb H, Alves C, Doan HA, Egele R, Assary RS, et al. Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-0tq7j
```

**BibTeX:**

```bibtex
@article{chowdhury2025automated,
  title={Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty},
  author={Chowdhury, Arindam and Harb, Hassan and Alves, Caio and Doan, Hieu Anh and Egele, Romain and Assary, Rajeev Surendran and Balaprakash, Prasanna},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-0tq7j}
}
```
