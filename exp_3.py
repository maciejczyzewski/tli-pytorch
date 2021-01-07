from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from research_models import get_list_of_models, get_model_debug, get_model_timm
from tli import get_tli_score


def eval_tls_score(debug):
    models = {}
    if debug:
        _models = [
            get_model_debug(seed=seed, classes=10, channels=3) for seed in range(5)
        ]
    else:
        _models = [
            get_model_timm(name=name, classes=10, channels=3)
            for name in get_list_of_models()
        ]

    for model in _models:
        print(f"----> {model.name}")
        models[model.name] = model

    arr_tli = []
    df_list, df_columns = [], ["name"]
    pbar = tqdm(models.items(), position=0, leave=True)
    for model_to_name, _model_to in pbar:
        pbar.set_description(
            "\x1b[6;30;42m" + f"MODEL {model_to_name} ".ljust(40, "=") + "\x1b[0m"
        )
        pbar.update()
        model_to = deepcopy(_model_to)
        print(f">>>>>>>>> model_to={model_to_name}")
        df_columns.append(model_to_name)
        df_row = [model_to_name]
        for model_from_name, model_from in tqdm(models.items(), position=0, leave=True):
            score = get_tli_score(model_from, model_to)
            print(f"\t {model_from_name:35} | score = {round(score, 4):10}")
            arr_tli.append(score)
            df_row.append(score)
        df_list.append(df_row)
    print("SCORE_TLI", np.mean(arr_tli))

    df = pd.DataFrame(df_list, columns=df_columns)
    df = df.set_index("name")
    return df


# FIXME: mozna zdefiniowac liste zamiast wszystkiego?
def experiment_3(debug=False, path=None):
    if not path:
        df = eval_tls_score(debug=debug)
        suffix = "" if debug == False else "_debug"
        df.to_csv(f"tmp/matrix{suffix}.csv", index=True)
    else:
        df = pd.read_csv(path)
        df = df.set_index("name")

    # order by algorithm name
    df = df.reindex(sorted(df.columns), axis=1)
    df.sort_index(inplace=True)

    df.loc["[MEAN]"] = df.mean()

    # FIXME: dla zabawy szybki treningu (a'la experiment)
    #           --------> szybki wykres --> bez "init classic"
    #           --------> (niech seed generuje losowe warstwy / conv2d)

    # FIXME: problem z latex --> trzeba wyescapowac te nazwy
    plt.style.use(["science", "ieee", "no-latex"])

    ax = plt.gca()
    ax.autoscale(tight=True)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    sns.heatmap(
        df,
        ax=ax,
        annot=True,
        cbar_kws={
            "shrink": 0.60,
            "pad": 0.025,
            "orientation": "horizontal",
            "label": "Similarity (TLI score)",
        },
        cmap="RdYlGn",
        linewidths=0,
        vmin=0,
        vmax=1,
        square=True,
        fmt=".1g",
    )

    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("[FROM]")
    ax.set_ylabel("[TO]")

    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.savefig("results/matrix.pdf")
