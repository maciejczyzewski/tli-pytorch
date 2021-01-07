from research_datasets import DATASET_CIFAR100
from research_models import get_model_timm
from research_plot import plot_key
from research_run import run_experiment
from research_utils import __lazy_init
from tli import apply_tli

MODEL_B0 = "efficientnet_b0"
MODEL_B1 = "efficientnet_b1"
MODEL_B2 = "efficientnet_b2"

# FIXME: add with "batchnorm" / without batchnorm --> option to `apply_tli`
# FIXME: ten sam eksperyment dla "duzy->maly" (inny kierunek)
# FIXME: ten sam eksperyment dla "bardzo rozne architektury" - niskie TLI

ITERATIONS = 1000

MODELS = {
    "B0": {"model": __lazy_init(get_model_timm, {"name": MODEL_B0})},
    "B1": {"model": __lazy_init(get_model_timm, {"name": MODEL_B1})},
    "B1_none": {"model": __lazy_init(get_model_timm, {"name": MODEL_B1})},
    "B2": {"model": __lazy_init(get_model_timm, {"name": MODEL_B2})},
    "B2_none": {"model": __lazy_init(get_model_timm, {"name": MODEL_B2})},
}

TRAIN_TLI_B0 = {
    "save_model": True,
    "model_name": "B0",
    "init": None,
    "meta": {"name": "B0 (ImageNet)", "color": "red", "shift": ITERATIONS * 0},
}

TRAIN_TLI_B1 = {
    "save_model": True,
    "model_name": "B1",
    "init": __lazy_init(apply_tli, {"teacher": "B0"}),
    "meta": {"name": "B1 + TLI(B0)", "color": "green", "shift": ITERATIONS * 1},
}

TRAIN_TLI_B1_NONE = {
    "model_name": "B1_none",
    "init": None,
    "meta": {
        "name": "B1 (ImageNet)",
        "color": "green",
        "shift": ITERATIONS * 1,
        "linestyle": "--",
    },
}

TRAIN_TLI_B2 = {
    "model_name": "B2",
    "init": __lazy_init(apply_tli, {"teacher": "B1"}),
    "meta": {"name": "B2 + TLI(B1)", "color": "blue", "shift": ITERATIONS * 2},
}

TRAIN_TLI_B2_NONE = {
    "model_name": "B2_none",
    "init": None,
    "meta": {
        "name": "B2 (ImageNet)",
        "color": "blue",
        "shift": ITERATIONS * 2,
        "linestyle": "--",
    },
}

EXP_2_CIFAR100 = {
    "dataset": DATASET_CIFAR100,
    "models": MODELS,
    "optimizer": {"name": "Adam", "lr": 0.003, "grad_acc": 8},
    "iterations": ITERATIONS,
    "loop": [
        TRAIN_TLI_B0,
        TRAIN_TLI_B1,
        TRAIN_TLI_B1_NONE,
        TRAIN_TLI_B2,
        TRAIN_TLI_B2_NONE,
    ],
}


def experiment_2(config_base):
    results = run_experiment(config_base)
    print(results)

    prefix = config_base["dataset"]["name"]

    # accuracy
    plot_key(
        results,
        key="acc_train",
        log=False,
        xlabel="iterations",
        ylabel="accuracy",
        title=prefix,
        subtitle="models",
        prefix="exp_2_" + prefix,
        ncol=1,
        # ylim=[0, 0.5],
    )

    # loss
    plot_key(
        results,
        key="loss_train",
        log=True,
        xlabel="iterations",
        ylabel="loss",
        title=prefix,
        subtitle="models",
        prefix="exp_2_" + prefix,
        ncol=1,
    )
