try:
    import google.colab

    google.colab.__init__ = "hello"
    IN_COLAB = True
except BaseException:
    IN_COLAB = False

try:
    import sys
    import IPython.core.ultratb

    sys.excepthook = IPython.core.ultratb.ColorTB()
except BaseException:
    pass

import collections
import json
import math
import multiprocessing
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# FIXME: how to download ImageNet?
from research_models import get_list_of_models
from research_utils import __lazy_init, create_hash
from tli import get_tli_score

try:
    # if you are hacker "dark_background"
    plt.style.use(["science", "ieee", "high-vis"])
except:
    print("try `pip install SciencePlots`")

################################################################################
# Config
################################################################################


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def call(cmd):
    print(
        f"\x1b[7;37;40m{cmd}\x1b[0m\n"
        + subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode(
            "utf-8"
        ),
        end="",
    )


call("mkdir -p tmp")
call("mkdir -p figures")

SEED = 666
set_seed(SEED)
cpu_count = multiprocessing.cpu_count()
torch.set_num_threads(cpu_count)
if sys.platform == "darwin":
    cpu_count = 0
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
warnings.simplefilter("ignore")

################################################################################
# Weights Initialization
################################################################################

# FIXME: move to `tli.py`?
class weights_init:
    def __init__(self, fn=None):
        self.fn = fn

    def f(self, layer):
        if hasattr(layer, "reset_parameters"):
            try:
                self.fn(layer.weight)
            except:
                print(f"ERROR: {layer}")
            if layer.bias is not None:
                try:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                        layer.weight
                    )
                except:
                    fan_in = 2
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(layer.bias, -bound, bound)


def apply_kaiming_uniform(model):
    model.apply(
        weights_init(
            fn=__lazy_init(torch.nn.init.kaiming_uniform_, {"a": math.sqrt(5)})
        ).f
    )
    return model


def apply_xavier_normal(model):
    model.apply(weights_init(fn=__lazy_init(torch.nn.init.xavier_normal_)).f)
    return model


################################################################################
# Train
################################################################################


class TrainTuple:
    def __init__(self, model, optimizer, dataset):
        self.model, self.optimizer, self.dataset = model, optimizer, dataset


def get_optimizer(model, **kwargs):
    if kwargs["name"] == "SGD":
        return optim.SGD(
            model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"]
        )
    if kwargs["name"] == "Adam":
        return optim.Adam(model.parameters(), lr=kwargs["lr"])
    raise Expection("optimizer not exists")


def test(model, test_loader, test_data):
    model.eval()
    test_loss, correct = 0, 0

    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in tqdm(test_loader, position=0, leave=True):
            data = data.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            # test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data)
    test_acc = correct / len(test_data)

    print(
        f"\033[94mFINAL\033[0m test_acc={test_acc:10} | test_loss={round(test_loss, 4)}"
    )
    return test_loss, test_acc


def train(model, optimizer, dataset, iterations=None):
    train_data, test_data = dataset()

    train_loader = DataLoader(
        train_data,
        batch_size=dataset["batch_size"],
        shuffle=True,
        num_workers=cpu_count,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=dataset["batch_size"],
        shuffle=False,
        num_workers=cpu_count,
        pin_memory=True,
    )

    scaler = torch.cuda.amp.GradScaler()
    _optimizer = get_optimizer(model, **optimizer)
    grad_acc = optimizer["grad_acc"] if "grad_acc" in optimizer else 16

    model.train()

    T1 = time.time()

    acc_train = []
    loss_train = []
    global_i, global_break = 0, False
    epochs = math.ceil(iterations / len(train_loader)) if iterations else 1
    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm(range(1, epochs + 1), position=0, leave=True)
    for epoch in pbar:
        header = f">>> epoch={epoch:5}"
        if iterations:
            header = f">>> epoch={epoch:5} | iterations={iterations:5}"

        pbar.set_description("\x1b[0;33;40m" + header + "\x1b[0m\n")
        pbar.update()

        for i, (data, target) in enumerate(tqdm(train_loader, position=0, leave=True)):
            data = data.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # gradient accumulation
            if i % grad_acc == 0 and i > 1:
                scaler.step(_optimizer)
                scaler.update()
                _optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
            loss = loss_fn(output, target)
            scaler.scale(loss / grad_acc).backward()
            loss_train.append(loss.item())

            pred = output.argmax(1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc_val = correct / dataset["batch_size"]
            acc_train.append(acc_val)

            if global_i % 100 == 0:
                print(
                    f" iter={global_i:5} loss={round(loss.item(), 4):6} acc={round(acc_val, 4):6}"
                )

            global_i += 1
            if iterations:
                if iterations <= global_i:
                    global_break = True
                    break

        if global_break:
            break

    # FIXME: option for `test` after i-th mod k iterations
    loss_test, acc_test = test(model, test_loader, test_data)

    T2 = time.time()
    print("TIME", T2 - T1)

    return global_i, loss_train, acc_train, loss_test, acc_test


################################################################################
# Results
################################################################################


class ResultPair:
    def __init__(self, config, data):
        self.config, self.data = config, data


def result_name(method, train_tuple):
    name = train_tuple.model.name
    dataset = train_tuple.dataset["name"]
    return f"{dataset}__model={name}__{method}"


def read_result(dhash):
    print(f"[dhash=\x1b[0;30;47m {dhash} \x1b[0m]", end=" ")
    path = f"tmp/{dhash}"
    if not os.path.isfile(path):
        print("run")
        return None
    blob = open(path, "r").read()
    data = json.loads(blob)
    print("cached")
    return data


def save_result(dhash, data=None):
    print("save", data.keys())
    blob = json.dumps(data)
    with open(f"tmp/{dhash}", "w") as file:
        file.write(blob)


################################################################################
# Runner
################################################################################


def run_experiment(config_base):
    models = {}
    print("\x1b[6;30;42m" + "MODELS".ljust(10, "=") + "\x1b[0m")
    for seed, (model_name, model_dict) in enumerate(config_base["models"].items()):
        print(f"creating `{model_name}`...")
        model_fn = model_dict["model"]
        models[model_name] = model_fn(
            classes=config_base["dataset"]["meta"]["classes"],
            channels=config_base["dataset"]["meta"]["channels"],
        )

        if "iterations" in model_dict:
            # FIXME: add option for custom "init"
            config_loop = {"model_name": model_name, "init": None}
            config = {**config_base, "this": config_loop}
            config["iterations"] = model_dict["iterations"]

            _, _, models = run_case(config, models, save_model=True)

    results = []
    pbar = tqdm(config_base["loop"], position=0, leave=True)
    for _config_loop in pbar:
        try_n = _config_loop["try_n"] if "try_n" in _config_loop else 1
        for try_i in range(1, try_n + 1):
            set_seed(SEED + try_i)
            config_loop = deepcopy(_config_loop)
            config_loop["try_i"] = try_i
            config = {**config_base, "this": config_loop}

            pbar.set_description("\x1b[6;30;42m" + "CASE".ljust(10, "=") + "\x1b[0m")
            pbar.update()

            out_config, data, models = run_case(config, models, save_model=False)
            print(out_config)
            results.append(ResultPair(out_config, data))

    def flatten(d, parent_key="", sep="__"):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def push_to_end(d, keys=None):
        _d_end = {}
        _d = copy(d)
        for key in keys:
            del d[key]
            _d_end[key] = _d[key]
        od = collections.OrderedDict({**d, **_d_end})
        return od

    df_list = []
    df_columns = None
    for i, result in enumerate(results):
        meta = copy(result.config["meta"])
        del result.config["meta"]
        blob_df = {**flatten(result.config), **result.data, "meta": meta}
        blob_df = push_to_end(blob_df, keys=["loss_train", "acc_train", "meta"])
        df_list.append(blob_df.values())
        if not df_columns:
            df_columns = blob_df.keys()

    today = datetime.today().strftime("%Y-%m-%d__%X")
    df = pd.DataFrame(df_list, columns=df_columns)
    df.to_csv(f"tmp/experiment__{today}.csv", index=False)

    # FIXME: check if collision? --> option to read `experiment.csv`
    print(";-)")

    return df


def run_case(config, models, save_model=False):
    data = {}

    del config["loop"]
    del config["models"]
    this = config["this"]
    del config["this"]

    if "save_model" in this and this["save_model"]:
        save_model = True

    iterations = config["iterations"] if "iterations" in config else None
    if "iterations" in this and this["iterations"]:
        iterations = this["iterations"]

    try_i = this["try_i"] if "try_i" in this else 1

    if try_i > 1 and save_model:
        # FIXME: zapisuje tylko try_i == 1 -> w innych wypadkach pomija?
        raise Exception("cannot use `try_n>1` with `save_model=True`")

    is_tli = True if (this["init"] and "TLI" in this["init"].name) else False
    init_name = this["init"].name if this["init"] else str(None)
    init_from = this["init"].params["teacher"] if is_tli else None

    print(f"[[[ dataset    = {config['dataset']['name']}")
    print(f"[[[ model_name = {this['model_name']} (save_model={save_model})")
    print(f"[[[ try_i      = {try_i} / is_tli={is_tli}")
    print(f"[[[ init       = {init_name} (init_from={init_from})")

    meta = this["meta"] if "meta" in this else {}
    print("META >>>", meta)

    model_name = this["model_name"]
    model_to_train = deepcopy(models[model_name])

    if torch.cuda.is_available():
        model_to_train.cuda()

    # FIXME: class for config?
    out_config = {
        "dataset": config["dataset"]["name"],
        "iterations": iterations,
        "this": {
            "model_name": this["model_name"],
            "init_name": init_name,
            "init_from": init_from,
            "try_i": try_i,
        },
    }

    _out_config = deepcopy(out_config)
    del _out_config["this"]["try_i"]
    ghash = create_hash(_out_config)
    out_config["ghash"] = ghash
    dhash = create_hash(out_config)
    out_config["dhash"] = dhash
    out_config["meta"] = meta
    path_to_model = f"tmp/models/{dhash}"

    data = read_result(dhash)

    if data:
        if save_model:
            print("loading cached model from file...")
            model_to_train.load_state_dict(
                torch.load(path_to_model, map_location=device)
            )
            models[model_name] = model_to_train

        return out_config, data, models

    if is_tli:
        model_teacher = models[init_from]
        this["init"].params["teacher"] = model_teacher

    if this["init"]:
        model_to_train = this["init"](model=model_to_train).to(device)

    train_tuple = TrainTuple(
        model=model_to_train, optimizer=config["optimizer"], dataset=config["dataset"]
    )

    iterations, loss_train, acc_train, loss_test, acc_test = train(
        train_tuple.model,
        train_tuple.optimizer,
        train_tuple.dataset,
        iterations=iterations,
    )

    data = {
        "iterations": iterations,
        "loss_train": loss_train,
        "acc_train": acc_train,
        "loss_test": loss_test,
        "acc_test": acc_test,
    }

    if save_model:
        call("mkdir -p tmp/models")
        torch.save(model_to_train.state_dict(), path_to_model)
        models[model_name] = model_to_train

    if is_tli:
        tli_score = get_tli_score(model_from=model_teacher, model_to=model_to_train)
        data["tli_score"] = tli_score
        print(f"[[[ TLI_SCORE={round(tli_score, 4)} ({init_from}->{model_name})")
    else:
        data["tli_score"] = None

    save_result(dhash, data=data)

    return out_config, data, models


################################################################################
# Experiments
################################################################################

# FIXME ---------- pomysly na eksperymenty...
# EXP 1: (use case) wiele zrodel -> jeden uczacy [2 wykresy + 2 tabelki]
#         FIXME -> wiecej roznych metod vs. np. ten laczacy (tli_auto)
# EXP 2: jedno zrodlo -> wiele uczacych [1 wykres + moze 1 wykres dla imagenet]
#         FIXME -> (1) zrobic prosta wersje (wraz z kaiming/xavier)
#                         --> pokaze oplacalnosc
#                  (2) zastanowic sie nad imagenet <-> imagenet
#### EXP 2: ---> moze by tak tabelke zrobic -->
####             nazwa modelu | normalny init | init wiedza | [[gain]]
#### EXP 2: ---> dla imagenet --> ta sama domena
# ----------------

if __name__ == "__main__":
    print("[run] running experiments")

    # FIXME: add flag for experiment run
    #        ---> automatic import from exp_*.py files
    #        ---> and params as __lazy_init

    # === list of models ===
    get_list_of_models()

    # === debug: only for testing framework ===
    from exp_debug import experiment_debug

    experiment_debug()

    # === EXP 1: basic use case ===
    # from exp_1 import experiment_1, EXP_1_CIFAR100
    # experiment_1(EXP_1_CIFAR100)

    # FIXME: [EXP 2]
    # from exp_2 import experiment_2, EXP_2_CIFAR100
    # experiment_2(EXP_2_CIFAR100)

    # === EXP 3: model matrix (tli score) ===
    # from exp_3 import experiment_3
    # experiment_3(path="matrix-similarity.csv")
    # experiment_3() # for generation

    # === EXP 4: batchnorm transfer ===
    # XXX: repeat but with "batchnorm" / on one plot?

    if IN_COLAB:
        call("zip latest_tmp.zip tmp/* tmp/**/**")
