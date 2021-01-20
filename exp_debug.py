from research_datasets import DATASET_MNIST
from research_models import get_model_debug
from research_plot import plot_key
from research_run import (apply_kaiming_uniform, apply_xavier_normal,
                          run_experiment)
from research_utils import __lazy_init
from tli import apply_tli

# FIXME: create_function for run --> "create_case"
#        z operatorem  "->" itp. oraz grupy (zamiast slownika?)

ITERATIONS = 1000 # 50

MODELS = {
    "hidden_A": {"model": __lazy_init(get_model_debug, {"seed": 1})},
    "hidden_B": {
        "model": __lazy_init(get_model_debug, {"seed": 2}),
        "iterations": ITERATIONS,
    },
    "BASE_MODEL": {"model": __lazy_init(get_model_debug, {"seed": 3})},
}

TRAIN_KAIMING_UNIFORM = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_kaiming_uniform),
    "meta": {"name": "kaiming uniform", "color": "gray",},
}

TRAIN_XAVIER_NORMAL = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_xavier_normal),
    "meta": {"name": "xavier normal", "color": "gray",},
}

TRAIN_TLI_1 = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "hidden_A"}),
    "meta": {"name": "TLI(hidden_A)", "color": "orange",},
}

TRAIN_TLI_2 = {
    "try_n": 2,
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "hidden_B"}),
    "meta": {"name": "TLI(hidden_B)", "color": "red",},
}

TRAIN_TLI_3 = {
    "model_name": "BASE_MODEL",
    "init": __lazy_init(apply_tli, {"teacher": "BASE_MODEL"}),
    "save_model": True,
    "meta": {"name": "transfer learning", "color": "black",},
}

EXP_MNIST = {
    "dataset": DATASET_MNIST,
    "models": MODELS,
    "optimizer": {"name": "Adam", "lr": 0.01, "grad_acc": 8},
    "iterations": ITERATIONS,
    "loop": [
        TRAIN_KAIMING_UNIFORM,
        TRAIN_XAVIER_NORMAL,
        TRAIN_TLI_1,
        TRAIN_TLI_2,
        TRAIN_TLI_3,
    ],
}


def experiment_debug():
    config_base = EXP_MNIST
    results = run_experiment(config_base)
    print(results)

    # FIXME: w pracy jako dwie kolumny???
    # FIXME: tylko MNIST + CIFAR10
    prefix = config_base["dataset"]["name"]
    # FIXME: `dataset param`

    # accuracy
    plot_key(
        results,
        key="acc_train",
        log=False,
        xlabel="iterations",
        ylabel="accuracy",
        title=prefix,
        prefix="debug_" + prefix,
    )

    # loss
    plot_key(
        results,
        key="loss_train",
        log=True,
        xlabel="iterations",
        ylabel="loss",
        title=prefix,
        prefix="debug_" + prefix,
    )

    ##############################################
    # sys.exit()
    # FIXME: zamiast accuracy plot --> zrobic to samo co w EXP_2
    #        albo wogole - zrezygnowac z tego figure!
    # plot_acc(results, boxplot=True, sort_by_rank=True, prefix=prefix)
    # FIXME: zaprezentowac wynik tabelka? (--> albo slupkowy?)
    # table_rank(results)
