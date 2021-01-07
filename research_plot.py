import functools
import operator

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def normalize_data(all_arr):
    loss_train_all = []
    loss_train_all2 = []
    for arr in all_arr:
        lx = len(arr)
        loss_train_all.append(arr)
        yhat = savgol_filter(arr, 2 + lx // 24 - ((lx // 24) % 2 == 0), 2)
        loss_train_all2.append(yhat)
    loss_train_all = np.array(loss_train_all)
    loss_train_all2 = np.array(loss_train_all2)
    lx = loss_train_all.shape[1]
    ylog = savgol_filter(
        np.log(loss_train_all.mean(axis=0)), lx // 2 - ((lx // 2) % 2 == 0), 2
    )
    yhat = savgol_filter(loss_train_all.mean(axis=0), lx // 2 - ((lx // 2) % 2 == 0), 2)
    return lx, loss_train_all, loss_train_all2, ylog, yhat


def plot_key(
    results,
    key,
    log=False,
    xlabel="x",
    ylabel="y",
    title=None,
    subtitle=None,
    prefix=None,
    ncol=3,
    ylim=None,
):
    print(f"[plot_loss] generation `{key}`")

    untex = lambda x: x.replace(" ", "\ ").replace("_", "\_")

    plt.clf()
    ax = plt.gca()
    ax.autoscale(tight=True)
    fig = plt.gcf()
    fig.set_size_inches(6, 2.5)  # (8, 2.5)
    ax.set_title(untex(title), loc="left")

    if log:
        norm = lambda x: np.log(x)
    else:
        norm = lambda x: x

    results_norm = []
    for _, result in results.groupby(results["ghash"]):
        lx, arr_all, arr_p50, arr_p90log, arr_p90 = normalize_data(result[key])
        results_norm.append(
            [result.iloc[0]["meta"], lx, arr_all, arr_p50, arr_p90log, arr_p90]
        )

    for _result in results_norm:
        meta, lx, arr_all, _, _, _ = _result

        shift = meta["shift"] if "shift" in meta else 0
        for i in range(arr_all.shape[0]):
            plt.plot(
                np.array(range(lx)) + shift,
                norm(arr_all[i]),
                color=meta["color"],
                alpha=0.05,
                linestyle="-",
            )

    for _result in results_norm:
        meta, lx, _, arr_p50, _, _ = _result

        shift = meta["shift"] if "shift" in meta else 0
        plt.fill_between(
            np.array(range(lx)) + shift,
            norm(arr_p50.mean(axis=0) - arr_p50.std(axis=0)),
            norm(arr_p50.mean(axis=0) + arr_p50.std(axis=0)),
            facecolor=meta["color"],
            alpha=0.4,
        )

    for _result in results_norm:
        meta, lx, _, _, arr_p90log, arr_p90 = _result

        label = untex(meta["name"])
        linestyle = meta["linestyle"] if "linestyle" in meta else "-"
        arr_fit = arr_p90log if log else arr_p90

        shift = meta["shift"] if "shift" in meta else 0
        plt.plot(
            np.array(range(lx)) + shift,
            arr_fit,
            color=meta["color"],
            alpha=1,
            linestyle=linestyle,
            label=label,
        )

    if log:
        ylabel = f"log({ylabel})"

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    # FIXME: dodac opcje / lub naprawic finalnie
    # plt.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    # ax.legend(handles, labels, loc="lower right",
    #          ncol=ncol)
    # ax.legend(handles, labels, loc="lower left",
    #           bbox_to_anchor=(0, -0.5), ncol=ncol)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)

    if subtitle:
        ax.text(
            1,
            1.025,
            untex(subtitle),
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    if ylim:
        ax.set_ylim(*ylim)

    prefix = f"{prefix}_" if prefix else ""
    plt.savefig(f"results/{prefix}_{key}.pdf")  # FIXME: add flag
    # FIXME: add pdf compression as flag to `run.py`


def table_rank(results):
    print("[table_rank] generation")

    results_norm = []
    for result in results:
        lx, arr_all, arr_p50, arr_p90log, arr_p90 = normalize_data(
            result.data, key="loss_train"
        )
        arr_acc_test = np.array([x["acc_test"] for x in result.data])
        arr_loss_test = np.array([x["loss_test"] for x in result.data])
        S1 = arr_acc_test.mean()
        S2 = arr_loss_test.mean()
        S3 = arr_p90[-1]
        # FIXME: Kullback–Leibler divergence
        # SF = S1 * (1 / S2) * (np.log1p(-S3))
        SF = S1  # * (1 / (1 + S2)) * (1 / (1 + S3))
        results_norm.append([SF, S1, S2, S3, result.config["name"]])

    # FIXME: table as ranking? row <- name + 3 sections (datasets) column
    #   name    |        MNIST        |       EMNIST      |     CIFAR10
    #              loss / acc / kl      loss / acc / kl    loss / acc / kl | ALL
    #                                  (combined score)    -----------------> ?

    results_norm = sorted(results_norm)[::-1]

    results_gain = []
    for b, a in zip(results_norm[::1], results_norm[1::1]):
        gain = round(100 * (b[0] - a[0]) / a[0], 2)
        results_gain.append(gain)
    results_gain.append(0)
    results_norm = [
        [results_gain[i]] + results_norm[i] for i in range(len(results_norm))
    ]

    print()
    from tabulate import tabulate

    headers = ["gain", "score", "acc_test", "loss_test", "loss_train", "method"]
    print(tabulate(results_norm, headers=headers))
    # FIXME: save to latex / figures (.tex files for \include)


def plot_acc(results, boxplot=False, sort_by_rank=True, prefix=None):
    print("[plot_acc] generation")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5))
    ax.autoscale(tight=True)
    ax.set_title(results[0].config["dataset"]["name"], loc="left")

    results_norm = []
    for result in results:
        label = result.config["name"].replace(" ", "\ ").replace("_", "\_")
        arr = np.array([x["acc_test"] for x in result.data])
        results_norm.append([arr.mean(), arr, label, result.config["color"]])

    if sort_by_rank:
        results_norm = sorted(results_norm, key=lambda x: x[0])[::-1]

    arr_all, labels, colors = [], [], []
    for _result in results_norm:
        arr_all.append(_result[1])
        labels.append(_result[2])
        colors.append(_result[3])

    plt.xticks(rotation=90)

    if boxplot:
        bplot = ax.boxplot(
            arr_all, notch=False, vert=True, patch_artist=True, labels=labels
        )

        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        margin = 0.2
        y_values = functools.reduce(operator.iconcat, arr_all, [])
        x_values = [y + 1 for y in range(len(arr_all))]
        v = np.array([box.get_path().vertices for box in bplot["boxes"]])
        xmin = v[:, :5, 0].min() - (max(x_values) - min(x_values)) * margin
        xmax = v[:, :5, 0].max() + (max(x_values) - min(x_values)) * margin
        ydiff = max(y_values) - min(y_values)
        ymin = min(y_values) - ydiff * margin
        ymax = max(y_values) + ydiff * margin
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
    else:
        bplot = ax.violinplot(arr_all, showmeans=False, showmedians=True)
        plt.setp(ax, xticks=[y + 1 for y in range(len(arr_all))], xticklabels=labels)

        for partname in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if partname not in bplot:
                continue
            vp = bplot[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1)

        for patch, color in zip(bplot["bodies"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        # FIXME: add margins

    # FIXME: debug here
    # ax.xaxis.set_label_position('top')
    # ax.set_xlabel('method')

    ax.yaxis.grid(True)
    ax.set_ylabel("accuracy")

    prefix = f"{prefix}_" if prefix else ""
    plt.savefig(f"results/{prefix}acc.png")


# FIXME: delete or repair?
def plot_sim(results, prefix=None):
    print("[plot_sim] generation")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.5))
    ax.autoscale(tight=True)
    ax.set_title(results[0].config["dataset"]["name"], loc="left")

    # FIXME: color?
    scatter_x, scatter_y, scatter_c = [], [], []
    plot_pair = {}

    for i, results_method in enumerate(results):
        method = results_method.config["name"]
        plot_pair[method] = []
        print("METHOD (line)", method)
        for result in results_method.data:
            print(
                "---->",
                "model",
                result["model_name"],
                "|",
                "acc_test",
                result["acc_test"],
                "|",
                "tli_score",
                result["tli_score"],
            )
            scatter_x.append(result["tli_score"])
            scatter_y.append(result["acc_test"])
            scatter_c.append(method)
            plot_pair[method].append([result["tli_score"], result["acc_test"]])
        print()

    def in_place_colors(arr_c):
        color_dict = {}
        cset = list(set(arr_c))
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(cset)))
        for i, _c in enumerate(arr_c):
            arr_c[i] = colors[cset.index(_c)]
            color_dict[_c] = arr_c[i]
        return arr_c, color_dict

    scatter_c, color_dict = in_place_colors(scatter_c)
    print(color_dict)

    plt.scatter(scatter_x, scatter_y, s=2, marker="o", c=scatter_c)

    for method in plot_pair:
        print(method)
        x, y = list(zip(*plot_pair[method]))
        label = method.replace(" ", "\ ").replace("_", "\_")
        plt.plot(
            np.unique(x),
            np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
            color=color_dict[method],
            label=label,
        )

    ax.yaxis.grid(True)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("similarity")

    # ax.set_xlim(0, 1) FIXME: debug
    ax.set_ylim(0, 1)

    # FIXME: problem with legend
    # plt.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    plt.legend()

    prefix = f"{prefix}_" if prefix else ""
    plt.savefig(f"results/{prefix}sim.pdf")


################################################################################

# FIXME: merge with `plot_key` (special param for "simple_plot")
def plot_abstract(
    results,
    key,
    log=False,
    xlabel="x",
    ylabel="y",
    title=None,
    subtitle=None,
    prefix=None,
    ncol=3,
    ylim=None,
):
    print(f"[plot_loss] generation `{key}`")

    untex = lambda x: x.replace(" ", "\ ").replace("_", "\_")

    plt.rcParams.update({"font.size": 16})

    plt.clf()
    ax = plt.gca()
    ax.autoscale(tight=True)
    fig = plt.gcf()
    fig.set_size_inches(6, 2.5)  # was: (8, 2.5)
    ax.set_title(untex(title), loc="left")

    if log:
        norm = lambda x: np.log(x)
    else:
        norm = lambda x: x

    results_norm = []
    for _, result in results.groupby(results["ghash"]):
        lx, arr_all, arr_p50, arr_p90log, arr_p90 = normalize_data(result[key])
        results_norm.append(
            [result.iloc[0]["meta"], lx, arr_all, arr_p50, arr_p90log, arr_p90]
        )

    for _result in results_norm:
        meta, lx, _, _, arr_p90log, arr_p90 = _result

        label = untex(meta["name"])
        linestyle = meta["linestyle"] if "linestyle" in meta else "-"
        arr_fit = arr_p90log if log else arr_p90

        shift = meta["shift"] if "shift" in meta else 0
        plt.plot(
            np.array(range(lx)) + shift,
            arr_fit,
            color=meta["color"],
            alpha=1,
            linestyle=linestyle,
            label=label,
            linewidth=3,
        )

    if log:
        ylabel = f"log({ylabel})"

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # FIXME: no legend! plot manually

    if subtitle:
        ax.text(
            1,
            1.025,
            untex(subtitle),
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    if ylim:
        ax.set_ylim(*ylim)

    prefix = f"{prefix}_" if prefix else ""
    plt.savefig(f"results/{prefix}_{key}.pdf")  # FIXME: add flag
    # FIXME: add pdf compression as flag to `run.py`
