# Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty
PyTorch implementation of Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty (https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/683e0487c1cb1ecda0ce5640/original/automated-learning-of-gnn-ensembles-for-predicting-redox-potentials-with-uncertainty.pdf)

## Overview
This library contains a PyTorch implementation of Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty as presented in [[1]](#citation).

## Dependencies

* **python>=3.10**
* **pytorch>=2.5**
* **pytorch-geometric>=2.6**
* **deephyper**
* **numpy**
* **pandas**
* **scikit-learn**
* **matplotlib**

## Structure
* [data_proc](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/src/data_proc.py): Code to convert the graph data into serialized data for non-GNN models.
* [Data](https://github.com/ArCho48/ion-complex/tree/master/Data): should contain the dataset (train_and_val_graph_list.pkl, test_graph_list.pkl,train_and_val_list.pkl, test_list.pkl, tmqm_list.pkl) The "_graph_" lists contain the PyG graphs for training and evaluating the GNN models while the remaining lists contain the serialized features without graph information for training and evaluating non-GNN models. 
* [hp_search](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/main.py): Main code for running the HPO loop. Run as *python3 hp_search.py* \[mode\]. Eg., to train GCN, run *python3 hp_search.py gcn*.
* [Training_](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/Training_.py): Code for trainign specific ML models. Ex., run as *python3 Training_gcn.py* for standalone GCN training. It is internally called by "hp_search".
* [ensemble_](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/main.py): Code for buidling model ensembles after HPO trials. Ex., to build GCN ensemble run as *python3 ensemble_gcn.py* 
* [inference](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/main.py): Code for running inference on specific models. Run as *python3 inference.py* \[mode\]. Eg., to infer on single GCN model, run *python3 inference.py gcn*.
* [models](https://github.com/ArCho48/Unrolled-WMMSE/tree/master/models): Stores trained models in folders with same name as the methods viz. "rf", "mlp", "svm" and "gcn".
* [results](https://github.com/ArCho48/Unrolled-WMMSE/tree/master/models): Stores results in folders with same names as the methods viz. "rf", "mlp", "svm" and "gcn".

## Usage

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:chowdhurya1@ornl.gov).

## Citation
```
[1] Chowdhury A, Harb H, Alves C, Doan HA, Egele R, Assary RS, et al. Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-0tq7j 
```

BibTeX format:
```
@article{chowdhury2025automated,
  title={Automated Learning of GNN Ensembles for Predicting Redox Potentials with Uncertainty},
  author={Chowdhury, Arindam and Harb, Hassan and Alves, Caio and Doan, Hieu Anh and Egele, Romain and Assary, Rajeev Surendran, Balaprakash, Prasanna},
  journal={chemarXiv preprint 10.26434/chemrxiv-2025-0tq7j},
  year={2025}
}

```