import pdb
import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set
from torch_geometric.nn import (
    GraphConv, SAGEConv, GATConv, GINConv, GINEConv,
    PNAConv, TransformerConv,
    global_add_pool, global_mean_pool, global_max_pool
)
from torch_geometric.nn.models import DimeNet, SchNet

from sklearn.ensemble import RandomForestRegressor,  ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Base class for GNN regression
class BaseGraphModel(nn.Module):
    def __init__(self, num_node_features, hp, edge_dim=None, set2set_steps=3):
        super().__init__()
        self.use_batch_norm = hp.get("use_batch_norm", False)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if self.use_batch_norm else None
        # hidden_dims and num_layers should be lists in hp, e.g. hidden_dim1, hidden_dim2
        hidden_dims = [hp.get(f"hidden_dim{i+1}", 64) for i in range(6)]
        num_layers = [hp.get(f"num_layers{i+1}", 2) for i in range(6)]
        self.set2set = Set2Set(hidden_dims[-1], processing_steps=set2set_steps)
        self._build_convs(num_node_features, hidden_dims, num_layers, hp, edge_dim)
        self.dense = nn.Linear(2*hidden_dims[-1], hidden_dims[-1])
        self.fc_out = nn.Linear(hidden_dims[-1], 1)

    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        # Example: build same conv type repeatedly, override in subclasses
        for i in range(num_layers[0]):
            conv = GraphConv(in_feat if i==0 else hidden_dims[0], hidden_dims[0])
            self.convs.append(conv)
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
        # sum_pool = global_add_pool(x, batch)
        # mean_pool = global_mean_pool(x, batch)
        # max_pool = global_max_pool(x, batch)
        s2s_pool   = self.set2set(x, batch)
        # x = torch.cat([sum_pool, mean_pool, max_pool], dim=1) #, s2s_pool
        x = torch.relu(self.dense(s2s_pool))#mean_pool))#x))
        return self.fc_out(x).squeeze(-1)

# Specific GNNs
class GCNNet(BaseGraphModel):
    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        for i, hid in enumerate(hidden_dims):
            for j in range(num_layers[i]):
                fin = in_feat if (i == 0 and j == 0) else (
                    hidden_dims[i-1] if j == 0 else hid
                )
                self.convs.append(GraphConv(fin, hid))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hid))

class SAGENet(BaseGraphModel):
    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        # in_feat += 3
        for i, hid in enumerate(hidden_dims):
            for j in range(num_layers[i]):
                fin = in_feat if (i == 0 and j == 0) else (hidden_dims[i-1] if j == 0 else hid)
                self.convs.append(SAGEConv(fin, hid, normalize=True, aggr=hp['aggr'], project=hp['project'], bias=hp['bias']))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hid))

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        # x = torch.cat([x,pos],dim=1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = torch.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
        # sum_pool = global_add_pool(x, batch)
        # mean_pool = global_mean_pool(x, batch)
        # max_pool = global_max_pool(x, batch)
        s2s_pool   = self.set2set(x, batch)
        # x = torch.cat([sum_pool, mean_pool, max_pool], dim=1) #, s2s_pool
        x = torch.relu(self.dense(s2s_pool))#mean_pool))#x))
        return self.fc_out(x).squeeze(-1)

class GATNet(BaseGraphModel):
    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        heads = hp.get("heads", 4)
        for i, hid in enumerate(hidden_dims):
            for j in range(num_layers[i]):
                fin = in_feat if (i == 0 and j == 0) else (hidden_dims[i-1] if j == 0 else hid)
                self.convs.append(GATConv(fin, hid // heads, heads=heads))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hid))

class GINNet(BaseGraphModel):
    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        for i, hid in enumerate(hidden_dims):
            for j in range(num_layers[i]):
                fin = in_feat if (i == 0 and j == 0) else (hidden_dims[i-1] if j == 0 else hid)
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(fin, hid),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hid, hid)
                )
                self.convs.append(GINConv(mlp))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hid))

class GINENet(nn.Module):
    def __init__(self, num_node_features, hp, edge_dim=None, set2set_steps=3):
        super().__init__()
        self.use_batch_norm = hp.get("use_batch_norm", False)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if self.use_batch_norm else None
        # hidden_dims and num_layers should be lists in hp, e.g. hidden_dim1, hidden_dim2
        hidden_dims = hp.get(f"hidden_dim1", 64)
        num_layers = hp.get(f"num_layers1", 2) 
        self.set2set = Set2Set(hidden_dims, processing_steps=set2set_steps)
        self._build_convs(num_node_features, hidden_dims, num_layers, hp, edge_dim)
        self.dense = nn.Linear(2*hidden_dims, hidden_dims)
        self.fc_out = nn.Linear(hidden_dims, 1)

    def _build_convs(self, in_feats, hidden_dims, num_layers, hp, edge_dim=None):
        for i in range(num_layers):
            nn_func = nn.Sequential(
                nn.Linear(in_feats if i==0 else hidden_dims, hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, hidden_dims)
            )
            self.convs.append(GINEConv(nn_func, edge_dim=edge_dim))
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))

    def forward(self, data):
        x, edge_attr, edge_feats, edge_index, batch = data.x, data.edge_attr, data.edge_feature, data.edge_index, data.batch
        edge_attr = torch.cat([edge_attr,edge_feats], dim=1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
        # sum_pool = global_add_pool(x, batch)
        # mean_pool = global_mean_pool(x, batch)
        # max_pool = global_max_pool(x, batch)
        s2s_pool   = self.set2set(x, batch)
        # x = torch.cat([sum_pool, mean_pool, max_pool], dim=1) #, s2s_pool
        x = torch.relu(self.dense(s2s_pool))#mean_pool))#x))
        return self.fc_out(x).squeeze(-1)

class PNANet(BaseGraphModel):
    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        aggregators = hp.get("aggregators", ["mean", "max", "sum"])
        pna_scalers = hp.get("pna_scalers", ["identity"])
        # Coerce from comma-separated strings if needed
        if isinstance(aggregators, str):
            aggregators = [s.strip() for s in aggregators.split(",") if s.strip()]
        if isinstance(pna_scalers, str):
            pna_scalers = [s.strip() for s in pna_scalers.split(",") if s.strip()]

        deg = hp.get("deg")

        for i, hid in enumerate(hidden_dims):
            for j in range(num_layers[i]):
                fin = in_feat if (i == 0 and j == 0) else (hidden_dims[i-1] if j == 0 else hid)
                self.convs.append(PNAConv(fin, hid, aggregators, pna_scalers, deg))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hid))

class TransformerNet(BaseGraphModel):
    def _build_convs(self, in_feat, hidden_dims, num_layers, hp, edge_dim=None):
        heads = hp.get("trans_heads", 4)
        for i, hid in enumerate(hidden_dims):
            for j in range(num_layers[i]):
                fin = in_feat if (i == 0 and j == 0) else (hidden_dims[i-1] if j == 0 else hid)
                self.convs.append(TransformerConv(fin, hid, heads=heads))
                if self.use_batch_norm:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hid))
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]*heads))

# DimeNet wrapper
class DimeNetNet(nn.Module):
    def __init__(self, num_node_features, hp, edge_dim=None):
        super().__init__()
        self.model = DimeNet(
            hidden_channels=hp.get("dimenet_hidden", 64),
            out_channels=1,
            num_blocks=hp.get("dimenet_blocks", 3),
            num_bilinear=hp.get("dimenet_bilinear", 4),
            num_radial=hp.get("dimenet_radial", 6),
            num_spherical=hp.get("dimenet_spherical", 7),
            max_num_neighbors=8,
            cutoff=3.5
        )

    def forward(self, data):
        # DimeNet expects data.z, data.pos, data.batch
        z = data.x[:,0].int()*118
        return self.model(z, data.pos, data.batch).squeeze(-1)

# SchNet wrapper
class SchNetNet(nn.Module):
    def __init__(self, num_node_features, hp, edge_dim=None):
        super().__init__()
        self.model = SchNet(
            hidden_channels=hp.get("schnet_hidden", 128),
            num_filters=hp.get("schnet_filters", 128),
            num_interactions=hp.get("schnet_interactions", 3)
        )

    def forward(self, data):
        # SchNet expects data.z, data.pos, data.batch
        z = 118*data.x[:,0].int()
        return self.model(z, data.pos, data.batch).squeeze(-1)

########################### Non-graph models #############################################

# MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hp):
        super(MLP, self).__init__()

        self.use_batch_norm = hp["use_batch_norm"]
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if self.use_batch_norm else None

        hidden_dims = [hp[f"hidden_dim{i+1}"] for i in range(6)]

        # Define the input layer
        self.convs.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        if self.use_batch_norm:
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[0]))

        # Define the hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if self.use_batch_norm:
                self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dims[i + 1]))

        # Define the output layer
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # Fully connected layers
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = torch.nn.functional.relu(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

        # Output layer
        x = self.fc_mu(x).squeeze(-1)

        return x

# Random Forest
def get_rf_model(hp) -> RandomForestRegressor:
    max_depth = hp['max_depth'] if (hp['max_depth'] is not None and hp['max_depth'] > 0) else None
    max_samples = hp['max_samples'] if hp['bootstrap'] else None
    max_features = 1.0 if hp['max_features'] == "all" else hp['max_features']

    if hp['splitter'] == "best":
        model = RandomForestRegressor(
            n_estimators=hp['n_estimators'],
            criterion=hp['criterion'],
            max_depth=max_depth,
            min_samples_split=hp['min_samples_split'],
            min_samples_leaf=hp['min_samples_leaf'],
            max_features=max_features,
            min_impurity_decrease=hp['min_impurity_decrease'],
            bootstrap=hp['bootstrap'],
            max_samples=max_samples,
            random_state=42,
        )
    elif hp['splitter'] == "random":
        model = ExtraTreesRegressor(
            n_estimators=hp['n_estimators'],
            criterion=hp['criterion'],
            max_depth=max_depth,
            min_samples_split=hp['min_samples_split'],
            min_samples_leaf=hp['min_samples_leaf'],
            max_features=max_features,
            min_impurity_decrease=hp['min_impurity_decrease'],
            bootstrap=hp['bootstrap'],
            max_samples=max_samples,
            random_state=42,
        )
    else:
        raise ValueError(
            f"The splitter is '{hp['splitter']}' when it should be in ['random', 'best']."
        )
    
    return model

# SVM
def get_svm_model(hp) -> SVR:
    model = SVR(
        kernel="poly", C=hp["C"], gamma=hp["gamma"], degree=hp["degree"]
    )
    return model

# xGBoost
def get_xgb_model(hp) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        max_depth=hp["max_depth"],
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=hp["early_stopping_rounds"],
    )
    return model
