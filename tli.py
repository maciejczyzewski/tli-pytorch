"""
weryfikacja:
- [ ] zrobic wizualizacje (reset -> applyied --> GT) dla KD
        --> SCORE/LOSS roznicy rozkladu --> SUMA/MEAN
        --> histogramy jako wrzuta do "folderu" dla warstwy
- [ ] wizualizacja dopasowania [1. matching 2. injection]
- [ ] zrobic exp__tli --> to samo co MNIST (1k)
          tylko modele 2flops moze 2 rozne od siebie
          jeden przeuczony drugi nie --> transfer --> patrzymy jaki score (ACC)
              [train/test mean]

technikalia:
- [ ] do kazdej warstwy dac "prawdopodobienstwo przypisania trans."
        te co maja najwyzsze prawd. (kilka) vs. (one) big boss
        to sa: rescale(X) + (wiekszawaga)*centercrop(X) + iter. mixing(zbioru)
- [ ] drzewiasty algo? similarity hashing? [[LHS]]
        jako dopasowanie!!!!!!!!!!!!!!!!!
- [ ] poczatkowe warstwy maja "wieksza wage"/"wieksze warstwy"
        --> zrobic jakas uczciwa krzywa z palca 100 -> 75 na ostatnich warstwach
- [ ] mixowanie wiele sieci z `results-imagenet`
            -> az ***nasycimy*** wszystkie wagi
- [ ] uzywanie `trace_graph` --> a nie "modulow" (uwzglednienie relacji)
- [ ] !!! UZYC graph cluster-ingu // zamiast DP

dodatki:
- [ ] FIXME: a co z reszta? np. ._bn0.num_batches_tracked
            ----------> model.state_dict().keys()
            zrobic cos w stylu --> with_meta = True
- [ ] FIXME: zrobic "szybkie" szukanie najlepszych modeli z ImageNet
        jesli ktos zdefiniuje [[auto=True]]
- [ ] analiza: https://github.com/KamilPiechowiak/weights-transfer/pull/17/files
- [ ] sprawdzic czy dziala [WS]/aug/grid tutaj?
- [ ] analiza: https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch/blob/master/README.md
- [ ] jakas porzadna nazwa np.
        yak shaving (use urban dictionary) // sponge function
                ---> still from crypto name / Unsponge ducktape
                ducktransfer
- [ ] analiza: https://github.com/MSeal/agglom_cluster
"""

# commit: dark tensor rises

import json
import random
import os
from collections import namedtuple
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
from graphviz import Digraph
from torch.autograd import Variable


################################################################################
# API
################################################################################

def get_model_timm(name="dla46x_c"):
    try:
        import timm
    except:
        raise Exception("timm package is not installed! try `pip install timm`")

    # FIXME: `channels`!!! and `classes`!!! as param (debug)
    model = timm.create_model(name, num_classes=10, in_chans=3, pretrained=True)
    return model


def str_to_model(name):
    if isinstance(name, str):
        print(f"loading `{name}` from pytorch-image-models...")
        model = get_model_timm(name)
    else:  # FIXME: check if "pytorch" model
        model = name
    return model


def apply_tli(model, teacher=None):
    model_teacher = str_to_model(teacher)
    transfer(model_teacher, model)
    return model

def get_tli_score(model_from, model_to):
    model_a = str_to_model(model_from)
    model_b = str_to_model(model_to)
    score_ab = transfer(model_a, model_b)
    score_ba = transfer(model_b, model_a)
    sim = (score_ab + score_ba) / 2
    print(
        f"[score_ab={round(score_ab, 2):6} score_ba={round(score_ba, 2):6} | sim={round(sim, 2):6}]"
    )
    return sim

# FIXME: remove?
def get_tli_score__old(model_from, model_to):
    # XXX: other info is here!
    model_a = str_to_model(model_from)
    model_b = str_to_model(model_to)
    score_A = transfer(model_a, model_a)
    score_B = transfer(model_b, model_b)
    score_T = transfer(model_a, model_b)
    # sim = score_T / min(score_A, score_B)
    sim = score_T / score_B  # min(score_A, score_B)
    # print("\033[92m=====================\033[0m")
    print(
        f"[score_a={round(score_A, 2):6} score_b={round(score_B, 2):6} score_t={round(score_T, 2):6} | sim={round(sim, 2):6}]"
    )
    # print("\033[92m=====================\033[0m")
    return sim

################################################################################
# Trace Graph
################################################################################


def apply_reset(model):
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return model


def resize_graph(dot, size_per_element=0.15, min_size=12):
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
    return size


def make_dot(var, params=None):
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )

    idx = random.randint(1, 100000)
    dot = Digraph(
        name=f"cluster_{idx}", node_attr=node_attr, graph_attr=dict(size="12,12")
    )
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    mod_op = ["AddBackward0", "MulBackward0", "CatBackward"]

    lmap, emap, hmap = {}, {}, {}

    def add_nodes(var, root=None, c=0, depth=0, branch=0, global_i=0):
        if var in seen:
            return None

        depth += 1
        global_i += 1

        if c not in lmap:
            lmap[c], emap[c] = [], []

        if root:
            emap[c].append((str(id(var)), root))

        # FIXME: move to function
        if hasattr(var, "variable"):
            u = var.variable
            name = param_map[id(u)] if params is not None else ""
            node_name = "%s\n %s" % (name, size_to_str(u.size()))
            lmap[c].append(
                {
                    "branch": branch,
                    "depth": depth,
                    "global_i": global_i,
                    "name": name,
                    "size": u.size(),
                    "type": "param",
                }
            )
            hmap[name] = str(id(var))
            dot.node(
                str(id(var)),
                f"c={c} branch={branch} depth={depth}\n" + node_name,
                fillcolor="lightblue",
            )
        else:
            node_name = str(type(var).__name__)
            if node_name in mod_op:
                depth = 0
                c = str(id(var))
                emap[c], lmap[c] = [], []
                dot.node(str(id(var)), node_name + f" [{c}]", fillcolor="green")
            else:
                lmap[c].append(
                    {
                        "branch": branch,
                        "depth": depth,
                        "global_i": global_i,
                        "name": node_name,
                        "type": "func",
                    }
                )
                dot.node(str(id(var)), node_name + f" [{c}]")
        seen.add(var)

        if hasattr(var, "next_functions"):
            for _branch, u in enumerate(var.next_functions):
                if node_name in mod_op:
                    branch = _branch
                if u[0] is not None:
                    dot.edge(str(id(u[0])), str(id(var)), color="blue")
                    add_nodes(
                        u[0],
                        root=str(id(var)),
                        c=c,
                        depth=depth,
                        branch=branch,
                        global_i=global_i,
                    )

    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    for c, edges in emap.items():
        with dot.subgraph(name=f"cluster_{idx}_{c}") as _c:
            _c.attr(color="blue")
            _c.attr(style="filled", color="lightgrey")
            _c.node_attr["style"] = "filled"
            _c.edges(edges)
            _c.attr(label=f"cluster {c}")

    return dot, lmap, hmap


def get_paths(lmap):
    pmap = {}
    for key, cluster in lmap.items():
        print(f"[[ {key} ]]")
        for block in cluster:
            if block["type"] == "param":
                print(f"\t name: \x1b[0;30;47m{block['name']}\x1b[0m")
                pmap[block["name"]] = cluster
            else:
                print(f"\t name: {block['name']}")
    print("PARAMS:", len(pmap))
    return pmap


################################################################################
# Visualization
################################################################################


def diff_graph(g1, gh1, g2, gh2, match, path="__tli_debug"):
    dot = Digraph(name="root")
    dot.graph_attr.update(compound="True")
    size_1 = resize_graph(g1)
    size_2 = resize_graph(g2)
    best_size = max(size_1, size_2)
    print(f"size_1 = {size_1} | size_2 = {size_2} | best_size = {best_size}")
    # config: size (FIXME: not changing?)
    g1.graph_attr.update(size=f"{best_size},{best_size}")
    g2.graph_attr.update(size=f"{best_size},{best_size}")
    # config: margins
    dot.graph_attr.update(rank="same", ranksep="5", nodesep="2", pad="2")
    # g1.graph_attr.update(compound="True")
    dot.subgraph(g2)
    dot.subgraph(g1)
    for name_from, pair in match.items():
        score, name_to = pair
        v = gh1[name_to]
        u = gh2[name_from]
        dot.edge(v, u, color="red", constraint="false", penwidth="5", weight="5")
    dot.render(filename=path + "__dot")
    os.system(f"rm {path}__dot")
    print("saved to file")


def show_graph(model, path="__tli_debug"):
    # FIXME: warning about 'torchviz'
    x = torch.randn(32, 3, 32, 32)
    v1, _, _ = make_dot(model(x), params=dict(model.named_parameters()))
    v1.render(filename=path + "__v1")
    os.system(f"rm {path}__v1")
    print("saved to file")


def get_graph(model):
    # FIXME: warning about 'torchviz'
    try:
        x = torch.randn(32, 3, 32, 32)
        g, lmap, hmap = make_dot(model(x), params=dict(model.named_parameters()))
    except:
        # FIXME: universal head? (what happens if MNIST?)
        print("ERROR: trying one channel")
        x = torch.randn(32, 1, 31, 31)
        g, lmap, hmap = make_dot(model(x), params=dict(model.named_parameters()))
    return g, lmap, hmap


################################################################################
# TLI
################################################################################


def fn_inject(from_tensor, to_tensor):
    from_slices, to_slices = [], []
    for a, b in zip(from_tensor.shape, to_tensor.shape):
        if a < b:
            from_slices.append(slice(0, a))
            to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
        elif a > b:
            from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
            to_slices.append(slice(0, b))
        else:
            from_slices.append(slice(0, a))
            to_slices.append(slice(0, b))
    to_tensor[tuple(to_slices)] = from_tensor[tuple(from_slices)]


_call_hash = lambda x: json.dumps(x, sort_keys=True)

# FIXME: cache? --> prefetch and hold in dict
def fn_get_tensor(key, d):
    tensor = None
    for _tensor in d:
        if key == _tensor["name"]:
            return _tensor
    return tensor


_fn_score_path = {}


def fn_score_path(a, b):
    _hash = "-".join(sorted([_call_hash(a), _call_hash(b)]))
    if _hash in _fn_score_path:
        return _fn_score_path[_hash]
    # FIXME: algorytm zwiazany z kolejnoscia! (teraz jest ignorowana)
    # FIXME: algortym zwiazany z edg-ami? (elementy moze sa te same)
    #                     ale czy tak samo polaczone? [jakis prosty O(n)]
    scores = []
    done = []
    for _a in a:
        max_score = 0
        max_i = 0
        # print("---->", _a['name'])
        for i, _b in enumerate(b):
            if i in done: # naive or 300iq?
                continue
            score = fn_score_tensor(_a, _b)
            if max_score < score:
                max_score = score
                max_i = i
            # print("\t", _b['name'], score)
        # XXX: print("\t FINAL -> ", _a['name'], max_score)
        scores.append(max_score)
        done.append(max_i)
    mean_score = np.mean(scores)
    # print("="*10 + " A "+ "="*10)
    # pprint(a)
    # print("="*10 + " B " +"="*10)
    # pprint(b)
    # print("MEAN SCORE", mean_score)
    _fn_score_path[_hash] = mean_score
    return mean_score


# FIXME: cache?
def fn_score_tensor(a, b):
    # FIXME: typ --> batchnorm / conv2d
    # FIXME: bias / weight? never match
    # FIXME: teorytycznie skad wiemy ze to inny branch?
    diff_branch = abs(a["branch"] - b["branch"])
    max_branch = 1 + max(a["branch"], b["branch"])
    diff_depth = abs(a["depth"] - b["depth"])
    max_depth = 1 + max(a["depth"], b["depth"])
    diff_global_i = abs(a["global_i"] - b["global_i"])
    max_global_i = 1 + max(a["global_i"], b["global_i"])
    _score_branch = diff_branch / max(20,max_branch)
    _score_depth = diff_depth / max(20,max_depth)
    _score_global_i = diff_global_i / max(20,max_global_i)
    red_flag = 1
    if a["type"] == b["type"] and a["type"] == "param":
        _score_size = 1
        for x, y in zip(a["size"], a["size"]):
            _score_size *= min(x / y, y / x)
        if len(a["size"]) != len(b["size"]):
            _score_size = 0
            red_flag = 1
        if ".bias" in a["name"] and ".weight" in b["name"]:
            _score_size = 0
            red_flag = 1
        if ".bias" in b["name"] and ".weight" in a["name"]:
            _score_size = 0
            red_flag = 1
    elif a["type"] == b["type"] and a["type"] == "func":
        _score_size = 0
        if a["name"] == b["name"]:
            _score_size = 1
    else:
        _score_size = 0
    # [balance]
    # print("sbranch", _score_branch)
    # print("sdepth", _score_depth)
    # print("ssize", _score_size)
    # FIXME: verify this approch
    score = red_flag * (
        (0 / 12) * (1 - _score_branch)
        + (4 / 12) * (1 - _score_depth)
        + (6 / 12) * _score_size
        + (2 / 12) * (1 - _score_global_i)
    )
    return score


def fn_match(paths_from, paths_to):
    scores = {}
    for a_key, a in paths_to.items():
        print(f"[[ {a_key} ]]")
        a_tensor = fn_get_tensor(a_key, a)
        partial_scores = []
        for b_key, b in paths_from.items():
            b_tensor = fn_get_tensor(b_key, b)
            score_path = fn_score_path(a, b)
            score_tensor = fn_score_tensor(a_tensor, b_tensor)
            # FIXME: debug? balance?
            score = (1/2)*score_path + (1/2)*score_tensor
            # score = score_path * score_tensor
            partial_scores.append([score, b_key])
            # print(f"\t --> {b_key:50} | score = {score:10}")
            # FIXME: sys.exit()
        best_fit = sorted(partial_scores)[::-1][
            0
        ]  # FIXME: get [top n] --> if not matched
        print(f"\t --> {best_fit[1]:50} | score = {best_fit[0]:10}")
        scores[a_key] = best_fit
    pprint(scores)
    return scores


def transfer(model_from, model_to, debug=False):
    print("[TRANSFER]")
    g1, lmap_from, hmap_from = get_graph(model_from)
    g2, lmap_to, hmap_to = get_graph(model_to)
    paths_from = get_paths(lmap_from)
    paths_to = get_paths(lmap_to)

    # FIXME: create dict tensor from `get_graph`
    scores = fn_match(paths_from, paths_to)

    p_from_ref = {}

    for name, param in model_from.named_parameters():
        p_from_ref[name] = param
    for name, param in model_to.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                name_fit = scores[name][1]
                fn_inject(p_from_ref[name_fit], param)

    partial_scores = []
    for _, match in scores.items():
        partial_scores.append(match[0])
    tli_score = np.mean(partial_scores)

    ### draw ###
    if debug:
        diff_graph(g1, hmap_from, g2, hmap_to, scores)

    print("PARAMS (from):", len(paths_from))
    print("PARAMS (  to):", len(paths_to))
    print("TLI SCORE:", tli_score)
    return tli_score


################################################################################
# Debug
################################################################################

if __name__ == "__main__":
    from research_models import get_model_debug

    model_debug = get_model_debug(seed=3, channels=3, classes=10)
    # model_b0 = get_model_timm("tf_mobilenetv3_small_minimal_100")
    # model_b0 = get_model_timm("regnetx_002")
    # model_b0 = get_model_timm("semnasnet_100")

    # model_A = get_model_timm("tf_mobilenetv3_small_minimal_100")
    # model_B = get_model_timm("tf_mobilenetv3_small_100")

    # [best pair for debug]
    model_A = get_model_timm("mixnet_s")
    model_B = get_model_timm("mixnet_m")

    show_graph(model_A)
    # show_graph(model_debug)

    # lmap = get_graph(model_A)
    # pmap = get_paths(lmap)

    transfer(model_A, model_B, debug=True) # tli sie

    ##########################################################
    # --> fn_stats() -> [abs.mean(), rozklad()]
    # --> fn_kullbeck(stats1, stats2) -> [0, 1]
    # FIXME: pretty list of modules? --> fn_stats
    #                                    if GT -> fn_kullbeck
    ##########################################################

    # print("A -> B")
    # x = get_tli_score(model_A, model_B)

    # print("B -> A")
    # x = get_tli_score(model_B, model_A)
