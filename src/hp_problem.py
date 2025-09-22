from deephyper.hpo import HpProblem

def define_hp_problem_ml(model_name):
    """
    Define a Deephyper HpProblem for the given model.
    """
    hp = HpProblem()
    # Common HPs
    if model_name == "MLP":
        hp.add_hyperparameter([False], "use_batch_norm")
        hp.add_hyperparameter((16, 64), "batch_size", default_value=32)
        hp.add_hyperparameter((3, 10), "patience", default_value=10)
        hp.add_hyperparameter((0.1, 0.9), "scheduler_factor", default_value=0.8)
        hp.add_hyperparameter((32, 512), "hidden_dim1", default_value=64)
        hp.add_hyperparameter((32, 512), "hidden_dim2", default_value=128)
        hp.add_hyperparameter((32, 512), "hidden_dim3", default_value=256)
        hp.add_hyperparameter((32, 512), "hidden_dim4", default_value=512)
        hp.add_hyperparameter((32, 512), "hidden_dim5", default_value=128)
        hp.add_hyperparameter((32, 512), "hidden_dim6", default_value=32)
        hp.add_hyperparameter((5e-4, 1e-2, "log-uniform"), "lr", default_value=0.0005)
        hp.add_hyperparameter([200], "epochs")
    elif model_name == "SVM":
        hp.add_hyperparameter((1, 100), "C", default_value=1)
        hp.add_hyperparameter(["scale", "auto"], "gamma", default_value="auto")
        hp.add_hyperparameter((1, 5), "degree", default_value=3)
    elif model_name == "RF":
        from ConfigSpace import (
            Constant,
            Categorical,
            ConfigurationSpace,
            Float,
            Integer,
            EqualsCondition,
        )
        config_space = ConfigurationSpace(
            name="sklearn.TreesEnsembleWorkflow",
            space={
                "n_estimators": Constant("n_estimators", value=2048),
                "criterion": Categorical(
                    "criterion",
                    items=["squared_error", "absolute_error", "friedman_mse"],
                    default="squared_error",
                ),
                "max_depth": Integer("max_depth", bounds=(0, 100), default=0),
                "min_samples_split": Integer(
                    "min_samples_split", bounds=(2, 50), default=2
                ),
                "min_samples_leaf": Integer("min_samples_leaf", bounds=(1, 25), default=1),
                "max_features": Categorical(
                    "max_features", items=["all", "sqrt", "log2"], default="sqrt"
                ),
                "min_impurity_decrease": Float(
                    "min_impurity_decrease", bounds=(0.0, 1.0), default=0.0
                ),
                "bootstrap": Categorical("bootstrap", items=[True, False], default=True),
                "max_samples": Float(
                    "max_samples", bounds=(1e-3, 1.0), default=1.0
                ),
                "splitter": Categorical(
                    "splitter", items=["random", "best"], default="best"
                ),
            },
        )
        config_space.add_condition(
            EqualsCondition(config_space["max_samples"], config_space["bootstrap"], True)
        )
        hp = HpProblem(config_space)
    elif model_name == "XGB":
        hp.add_hyperparameter([2048], "n_estimators"),
        hp.add_hyperparameter((5e-3, 5e-2, "log-uniform"), "learning_rate", default_value=0.01),
        hp.add_hyperparameter((0, 100), "max_depth", default_value=6),
        hp.add_hyperparameter(["rmse", "mae"], "eval_metric", default_value="rmse"),
        hp.add_hyperparameter((10, 100), "early_stopping_rounds", default_value=50)
    else:
        raise ValueError(f"HP not defined for model: {model_name}")
    return hp

def define_hp_problem_graph(model_name):
    """
    Define a Deephyper HpProblem for the given model.
    Supported model_names (case-insensitive):
      GIN, SAGE, GAT, GINE, PNA, Transformer, DimeNet, SchNet
    """

    hp = HpProblem()

    # --------------------
    # Common HPs (shared)
    # --------------------
    hp.add_hyperparameter([True, False], "use_batch_norm", default_value=True)
    hp.add_hyperparameter((16, 64), "batch_size", default_value=48)
    hp.add_hyperparameter((5, 10), "patience", default_value=10)
    hp.add_hyperparameter((0.1, 0.9), "scheduler_factor", default_value=0.62)
    hp.add_hyperparameter((1e-5, 1e-3, "log-uniform"), "lr", default_value=9.936e-4)
    hp.add_hyperparameter([200], "epochs")

    def add_blocked_gnn_hps():
        hp.add_hyperparameter((16, 256), "hidden_dim1", default_value=128)
        hp.add_hyperparameter((16, 256), "hidden_dim2", default_value=128)
        hp.add_hyperparameter((16, 256), "hidden_dim3", default_value=128)
        hp.add_hyperparameter((16, 256), "hidden_dim4", default_value=128)
        hp.add_hyperparameter((16, 256), "hidden_dim5", default_value=128)
        hp.add_hyperparameter((16, 256), "hidden_dim6", default_value=128)

        # per-block repeat counts 
        hp.add_hyperparameter((1, 3), "num_layers1", default_value=1)
        hp.add_hyperparameter((1, 3), "num_layers2", default_value=1)
        hp.add_hyperparameter((1, 3), "num_layers3", default_value=1)
        hp.add_hyperparameter((1, 3), "num_layers4", default_value=1)
        hp.add_hyperparameter((1, 3), "num_layers5", default_value=1)
        hp.add_hyperparameter((1, 3), "num_layers6", default_value=1)

    # --------------------
    # Model-specific HPs
    # --------------------
    if model_name == "GIN" or model_name == "GCN":
        add_blocked_gnn_hps()

    elif model_name == "SAGE":
        # GraphSAGE: same blocked dims/layers as GIN
        add_blocked_gnn_hps()
        hp.add_hyperparameter(["mean", "max"], "aggr", default_value="mean")
        hp.add_hyperparameter([True, False], "project", default_value=True)
        hp.add_hyperparameter([True, False], "bias", default_value=True)

    elif model_name == "GAT":
        # GAT: same blocked dims/layers + attention heads
        add_blocked_gnn_hps()
        hp.add_hyperparameter([2, 4, 8], "heads", default_value=4)

    elif model_name == "GINE":
        # GINE: single hidden width repeated 'num_layers1' times
        hp.add_hyperparameter((16, 256), "hidden_dim1", default_value=128)
        hp.add_hyperparameter((1, 6), "num_layers1", default_value=3)

    elif model_name == "PNA":
        # PNA: same blocked dims/layers + aggregator/scaler choices
        add_blocked_gnn_hps()
        # Aggregators and scalers are lists; offer a few curated options
        hp.add_hyperparameter(
            [
                ["mean", "max", "sum"],
                ["mean", "max", "sum", "min"],
                ["mean", "max", "sum", "var"],
            ],
            "aggregators",
            default_value=["mean", "max", "sum"],
        )
        hp.add_hyperparameter(
            [
                ["identity"],
                ["identity", "amplification", "attenuation"],
            ],
            "pna_scalers",
            default_value=["identity"],
        )
        # 'deg' is expected from data preprocessing

    elif model_name == "Transformer":
        # Graph TransformerConv: blocked dims/layers + head count
        add_blocked_gnn_hps()
        hp.add_hyperparameter([2, 4, 8], "trans_heads", default_value=4)

    elif model_name == "DimeNet":
        # DimeNet-specific knobs
        hp.add_hyperparameter((32, 256), "dimenet_hidden", default_value=64)
        hp.add_hyperparameter((2, 6), "dimenet_blocks", default_value=6)
        hp.add_hyperparameter((2, 8), "dimenet_bilinear", default_value=8)
        hp.add_hyperparameter((4, 16), "dimenet_radial", default_value=6)
        hp.add_hyperparameter((2, 8), "dimenet_spherical", default_value=2)

    elif model_name == "SchNet":
        # SchNet-specific knobs
        hp.add_hyperparameter((64, 256), "schnet_hidden", default_value=128)
        hp.add_hyperparameter((64, 256), "schnet_filters", default_value=128)
        hp.add_hyperparameter((2, 6), "schnet_interactions", default_value=3)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return hp
