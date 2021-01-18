from research_datasets import DATASET_CIFAR100
from research_models import get_model_timm
from research_plot import plot_abstract, plot_key
from research_run import (apply_kaiming_uniform, apply_xavier_normal,
                          run_experiment)
from research_utils import __lazy_init
from tli import apply_tli

# FIXME: dodac flage na wyrenderowanie .pdf / font-a latex
# FIXME: zrobic tak jak w weight standraization --> BAR PLOT? taki chamski
# FIXME: dodac renderowanie tabelki w latex!

MODEL_BASE = "mnasnet_100"
MODEL_TLI_90 = "spnasnet_100"
MODEL_TLI_10 = "tf_efficientnet_b0_ap"

MODELS = {
    "BASE_MODEL": {"model": __lazy_init(get_model_timm, {"name": MODEL_BASE})},
    "hidden_90_imagenet": {
        "model": __lazy_init(get_model_timm, {"name": MODEL_TLI_90})
    },
    "hidden_90": {
        "model": __lazy_init(get_model_timm, {"name": MODEL_TLI_90}),
        "iterations": 10000,
    },
    "hidden_10_imagenet": {
        "model": __lazy_init(get_model_timm, {"name": MODEL_TLI_10})
    },
    "hidden_10": {
        "model": __lazy_init(get_model_timm, {"name": MODEL_TLI_10}),
        "iterations": 10000,
    },
}

TRAIN_KAIMING_UNIFORM = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_kaiming_uniform),
    "meta": {"name": "(init) kaiming uniform", "color": "gray",},
}

TRAIN_XAVIER_NORMAL = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_xavier_normal),
    "meta": {"name": "(init) xavier normal", "color": "gray",},
}

TRAIN_TLI_90_IMAGENET = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "hidden_90_imagenet"}),
    "meta": {
        "name": "TLI(score=0.9; ImageNet)",
        "color": "darkgreen",
        "linestyle": "--",
    },
}

TRAIN_TLI_90 = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "hidden_90"}),
    "meta": {"name": "TLI(score=0.9; CIFAR100)", "color": "green",},
}

TRAIN_TLI_10_IMAGENET = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "hidden_10_imagenet"}),
    "meta": {"name": "TLI(score=0.1; ImageNet)", "color": "darkred", "linestyle": "--"},
}

TRAIN_TLI_10 = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "hidden_10"}),
    "meta": {"name": "TLI(score=0.1; CIFAR100)", "color": "red",},
}

TRAIN_IMAGENET = {
    "model_name": "BASE_MODEL",
    "init": None,
    "meta": {
        "name": "transfer learning; ImageNet",
        "color": "black",
        "linestyle": "--",
    },
}

EXP_1_CIFAR100 = {
    "dataset": DATASET_CIFAR100,
    "models": MODELS,
    "optimizer": {"name": "Adam", "lr": 0.003, "grad_acc": 8},
    "iterations": 5000,
    "loop": [
        TRAIN_KAIMING_UNIFORM,
        TRAIN_XAVIER_NORMAL,
        TRAIN_TLI_90_IMAGENET,
        TRAIN_TLI_90,
        TRAIN_TLI_10_IMAGENET,
        TRAIN_TLI_10,
        TRAIN_IMAGENET,
    ],
}


def experiment_1(config_base):
    # FIXME: "meta" overlap --> tak abym mogl nowy wykres zdefiniowac
    #        --> moge sobie modyfikowac z tego poziomu dane
    #        --> filtrowac / laczyc z zapamietanymi
    results = run_experiment(config_base)
    print(results)

    prefix = config_base["dataset"]["name"]

    plot_abstract(
        results,
        key="acc_train",
        log=False,
        xlabel="iterations",
        ylabel="accuracy",
        title=prefix,
        subtitle=MODEL_BASE,
        prefix="abstract_" + prefix,
        ncol=1,
        ylim=[0, 0.5],
    )
    sys.exit()

    # accuracy
    plot_key(
        results,
        key="acc_train",
        log=False,
        xlabel="iterations",
        ylabel="accuracy",
        title=prefix,
        subtitle=MODEL_BASE,
        prefix="exp_1_" + prefix,
        ncol=1,
        ylim=[0, 0.5],
    )

    # loss
    plot_key(
        results,
        key="loss_train",
        log=True,
        xlabel="iterations",
        ylabel="loss",
        title=prefix,
        subtitle=MODEL_BASE,
        prefix="exp_1_" + prefix,
        ncol=1,
    )
