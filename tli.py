# MUSIC: https://www.youtube.com/watch?v=F3OFIuIXcSo
#        https://www.youtube.com/watch?v=m_ysN9BQm8s
# FIXME: https://arxiv.org/pdf/2006.12986.pdf
# -----> depth/width/kernel level --> (podzial kodu)
# https://github.com/JaminFong/FNA/blob/master/fna_det/tools/apis/param_remap.py
# (paper) Karate Club
# https://arxiv.org/pdf/2003.04819.pdf

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
- [ ] wielopoziomowe dopasowanie/clustry (a nie standaryzacja):
        (zagniezdzone clustry)
    - in/mid/out -> block -> branch -> grupa tensorow -> tensor -> itp.
"""

# commit: dark tensor rises

import collections
import os
import random
import sys
# FIXME: repair config "reinit" case
from copy import copy
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from graphviz import Digraph
from karateclub import FeatherNode, NetMF
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

################################################################################
# API
################################################################################


def apply_tli(model, teacher=None):
    # print(f"[TLI]   model={model}")
    # print(f"[TLI] teacher={teacher}")
    model_teacher = str_to_model(teacher)
    transfer(model_teacher, model)
    return model


def get_tli_score(model_from, model_to):
    model_a = str_to_model(model_from)
    model_b = str_to_model(model_to)
    sim, _, _, _ = transfer(model_a, model_b)
    return sim


def get_model_timm(name="dla46x_c"):
    try:
        import timm
    except:
        raise Exception("timm package is not installed! try `pip install timm`")

    # FIXME: `channels`!!! and `classes`!!! as param (debug)
    model = timm.create_model(name, num_classes=10, in_chans=3, pretrained=True)
    return model


# FIXME: move to class ModelObj
def str_to_model(name):
    if isinstance(name, str):
        print(f"loading `{name}` from pytorch-image-models...")
        model = get_model_timm(name)
    else:  # FIXME: check if "pytorch" model
        model = name
    return model


# def get_tli_score(model_from, model_to):
#     model_a = str_to_model(model_from)
#     model_b = str_to_model(model_to)
#     score_ab = transfer(model_a, model_b)
#     score_ba = transfer(model_b, model_a)
#     sim = (score_ab + score_ba) / 2
#     print(
#         f"[score_ab={round(score_ab, 2):6} score_ba={round(score_ba, 2):6} | sim={round(sim, 2):6}]"
#     )
#     return sim


################################################################################
# Utils
################################################################################


def apply_hard_reset(model):
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return model


def fn_inject(from_tensor, to_tensor):
    # FIXME: debug -> vis. -> rescale
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


################################################################################
################################################################################
################################################################################

# FIXME: ladnie podzielic na "matching" / "injection"

##########################################################
# --> fn_stats() -> [abs.mean(), rozklad()]
# --> fn_kullbeck(stats1, stats2) -> [0, 1]
# FIXME: pretty list of modules? --> fn_stats
#                                    if GT -> fn_kullbeck
##########################################################

# DIST: https://github.com/timtadh/zhang-shasha
# "graph matching" https://arxiv.org/pdf/1904.12787.pdf
# https://github.com/deepmind/deepmind-research/tree/master/graph_matching_networks
# Graph / Node [Embedding???] / Graph2Vec
# https://github.com/Jacobe2169/GMatch4py
###############
# https://github.com/benedekrozemberczki/awesome-graph-classification
# WeisfeilerLehman ??????

################# BEST ###################
## https://karateclub.readthedocs.io/en/latest/notes/introduction.html
## SELF-LEARN? ---> weak-estimators???

# https://github.com/topics/graph2vec

# [FIXME] konstytucja rewolucjonisty
# kazdy blok ma: [[ a) pozycje b) strukture c) rozmiar ]]
# taki musi byc tez *score*

# FIXME: matching blokow to prekalkulacja scorow
# --> rozwazania beda tensor vs tensor

# STEPS:
# (1) (s-match) cluster vs cluster (top k/percentile)
# (2) (d-match) cluster (iterate -> tensor // double in-tree/out-tree)
# (3) (w-inject)/(k-inject) combo-inject

# pozycja -> level
# struktura -> treedist(edges)

# meta-learning?

#       a) pozycje (normalized)
#       b) strukture (graph features)
#       c) rozmiar (tensor)

# AS DIFF (a -> b)
# [a] (position, structure) -
# [b] (position, structure)

# SELF-LEARN / [[[SELF-ENCODER?]]]

# AS UNSUPERVISED? --> data augmentation / [permutation]??????
#                  --> rozne tensory z tego modulu sa?????????
# [a] [(position, structure), (tensor)] --> 0/1

# label: y = [1 -> to ten tensor], [0 -> to nie ten tensor]

# GENIUS =========
# self-learn? permutacje? graph2vec na samym sobie?
# uczy sie rozpoznawac jaki to tensor xd
# https://arxiv.org/pdf/2010.12878.pdf
# ================
# ??????????????
# http://ryanrossi.com/pubs/KDD18-graph-attention-model.pdf

# FIXME: "faster and naive alternative"
# after scoring [size] -->> [maximum weight bipartite b-matching]

########################## DRAFT ###############################################

# FIXME: [ALWAYS FIT ALL?] model.fit()
# FIXME: [ALWAYS in-tree/out-tree split]
# FIXME: graph embeddings? from all graphs? (EVEN student+teacher)
#                or per `cluster` --> graph embedding? (1)
#               and per `graph` --> graph embedding? (2)
#                      so i have then 2 different embeddings space...

# N basic    --> (nodeE, shape)
# N advanced --> (nodeE, graphE in-tree, graphE out-tree, shape)
# FIXME: there is a problem with "NODE Embedding"?
#                --> change to split in/out graph embedding?

# [S + G + N], [N_i] ---> 0/1 [smooth labeling 0.5/0.25 itc.]
# 1) structure (graph embedding)
# 2) graph (graph embedding)
# 3) node (node embedding)

# ---> laczy na chwile graf [student-teacher] --> cos na tym robi?
# ---> duzo prostsze featury // --> l / max(l) --> c / max(c)

# [XXX] READ THIS: https://markheimann.github.io/projects.html
# https://sci-hub.se/https://link.springer.com/chapter/10.1007/978-3-319-93040-4_57

# print(clf.predict(predictionData),'\n')

# model = LinearRegression().fit(X_train, y_train)
# print(model)

# y_hat = downstream_model.predict_proba(X_test)[:, 1]
# auc = roc_auc_score(y_test, y_hat)
# print('AUC: {:.4f}'.format(auc))

############################################################################

# FIXME: BIAS / WEIGHT (wildcard)
# FIXME: split_d for `student` then ensemble for encoder?

# >>> FOR FLOW
# split_map = split_flow_level(graph_teacher)
# pprint(split_map)
# encoded_split_map = encoder_graph(split_map)
# pprint(encoded_split_map)

# >>> FOR CLUSTERS
# for cluster_idx in graph_teacher.cluster_map.keys():
#     split_map = split_cluster_level(graph_teacher, cluster_idx)
#     pprint(split_map)
#     print(f"cluster_idx={cluster_idx}")
#     break

# >>> ALL FOR NODES
# edges = []
# for a, dst in graph_teacher.edges.items():
#     for b in dst:
#         edges.append([a, b])
# obj = encoder_nodes(edges)
# pprint(obj)
# sys.exit(1)

# >>> FOR NODES IN CLUSTER
# for cluster_idx in graph_teacher.cluster_map.keys():
#     obj = encoder_nodes(graph_teacher.cluster_map[cluster_idx].edges)
#     pprint(obj)
#     print("="*30)

# FIXME: KD-tree?
# FIXME: zrobic wizualizacje matchingu!!!!!!!!!!!!!!!!!!!!!!!
#     (przetestowac laczac ze soba 2 tensory)
#     (dodatek - wizualizacja dodatkowych `edges` do debugu)

#### [[[[[Fast Network Alignment]]]]]]] / xNetMF

# XXX XXX XXX XXX XXX [READ THIS] #######################
# https://gemslab.github.io/papers/heimann-2018-regal.pdf
# https://github.com/GemsLab/REGAL
#########################################################

# class NodeFeatures
#   [a] structures_info
#   [b] graph_info
#   [c] ???? shape
# for multiple matches [[ SparseMAP ]]
# ---> https://arxiv.org/pdf/1802.04223.pdf

# KD-tree? for representations?
# ----> MATRIX???

# matching if provided map


################################################################################
################################################################################
################################################################################

# XXX XXX XXX XXX XXX [READ THIS] #######################
# https://gemslab.github.io/papers/heimann-2018-regal.pdf
# https://github.com/GemsLab/REGAL
#########################################################


def get_networkx(edges, dag=True):
    if dag:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges)
    return G


def show_networkx(graph):
    if isinstance(graph, list):
        graph = get_networkx(edges=graph)
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(graph, pos, with_labels=True, arrows=True)
    plt.show()


def dag_split(edges, token, root=None):
    graph = {}
    for a, b in edges:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
    edges_split = []
    visited, queue = set(), collections.deque([root])
    while queue:
        stop = False
        node_root = queue.popleft()
        if node_root not in graph:
            continue
        if node_root == token:
            break
        for node in graph[node_root]:
            if node not in visited:
                if node == token:
                    stop = True
                edges_split.append([node_root, node])
                visited.add(node)
                queue.append(node)
        if stop:
            break
    # FIXME: empty graphs?
    if not edges_split:
        edges_split.append([token, token])
    return edges_split


def graph_splits(edges, nodes=False):
    G = get_networkx(edges)
    order = list(nx.topological_sort(G))
    if len(order) == 0:
        return {}
    idx_src, idx_dst = order[0], order[-1]
    if not nodes:
        nodes = set()
        for a, b in edges:
            nodes.add(a)
            nodes.add(b)
    split_map = {}
    for idx in nodes:
        in_tree = dag_split(edges, idx, root=idx_src)
        out_tree = dag_split(edges, idx, root=idx_dst)
        split_map[idx] = {"in-tree": in_tree, "out-tree": out_tree}
    return split_map


def graph_norm(edges, attr=None):
    normal_id_map = {}
    normal_id_iter = [0]
    rev_mask = {}

    def __for_single(idx):
        if not idx in normal_id_map:
            normal_id_map[idx] = normal_id_iter[0]
            rev_mask[normal_id_iter[0]] = idx
            normal_id_iter[0] += 1

    # random.shuffle(edges)

    for a, b in edges:
        __for_single(a)
        __for_single(b)

    norm_edges = []
    for a, b in edges:
        norm_edges.append([normal_id_map[a], normal_id_map[b]])

    # norm_edges = sorted(norm_edges)

    norm_attr = []
    if attr:
        for i in range(len(normal_id_map.keys())):
            norm_attr.append(attr[rev_mask[i]])

    return norm_edges, rev_mask, norm_attr


def utils_map_to_mask(split_map):
    mask, graphs = [], []
    for key, split_dict in split_map.items():
        for dict_key in split_dict.keys():
            _g, rev_mask, _ = graph_norm(split_dict[dict_key])
            g = get_networkx(_g, dag=False)
            mask.append([key, dict_key])
            graphs.append(g)
    return mask, graphs


def utils_mask_to_map(mask, X):
    split_map = {}
    for i, (key, dict_key) in enumerate(mask):
        if key not in split_map:
            split_map[key] = {}
        split_map[key][dict_key] = X[i]
    return split_map


################################################################################


def split_flow_level(graph):
    edges = []
    for edge in graph.cluster_links:
        cluster_idx_1 = graph.nodes[edge[0]].cluster_idx
        cluster_idx_2 = graph.nodes[edge[1]].cluster_idx
        edges.append([cluster_idx_1, cluster_idx_2])
    return graph_splits(edges)


def split_cluster_level(graph, cluster_idx):
    edges = graph.cluster_map[cluster_idx].edges
    return graph_splits(edges)


def encode_graph(split_map):
    mask, graphs = utils_map_to_mask(split_map)

    # FIXME: move to settings
    from karateclub import GL2Vec

    model = GL2Vec(dimensions=16) #FeatherGraph(eval_points=2, order=2)
    print("FIT")
    model.fit(graphs)
    print("EMBEDDING")
    X = model.get_embedding()
    print("-------------------->", X.shape)

    return utils_mask_to_map(mask, X)


################################################################################
# TLI
################################################################################


class TLIConfig(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

from karateclub import Diff2Vec
embedding_dim = 5  # best 4, 6, 5 / FIXME: was 9, how to find?
CONFIG = TLIConfig(
    {
        # FIXME: move outsite? --> lazy_load?
        "node_embedding_attributed": FeatherNode( # 2, 4
            eval_points=4, order=4, svd_iterations=100, reduction_dimensions=32
        ),
        "node_embedding_neighbourhood": NetMF(
             dimensions=embedding_dim
        ),  # FIXME: use xNetMF
                # Diff2Vec(diffusion_number=5, diffusion_cover=5, dimensions=embedding_dim),
        "autoencoder": MLPRegressor(
            max_iter=100, # 100 // 3,  # FIXME: best 50
            early_stopping=False,
            activation="relu",
            solver="adam",
            tol=0.0001,
            ##############################################
            # n_iter_no_change=100, # FIXME: is that good?
            ##############################################
            hidden_layer_sizes=(200, 50, 25,),  # 125, 25
            warm_start=True,
            learning_rate_init=0.0005,
            alpha=0.001,
            verbose=True,
        ),
        "test_size": 0.05,  # FIXME: this is important!
        "samples_per_tensor": 10,
    }
)


def E_nodes(edges, attr=None):
    norm_graph, rev_mask, norm_attr = graph_norm(edges, attr=attr)

    if len(rev_mask) == 0:
        return []

    model = (
        CONFIG.node_embedding_attributed
        if attr
        else CONFIG.node_embedding_neighbourhood
    )

    graph = get_networkx(norm_graph, dag=False)
    if attr:
        model.fit(graph, np.array(norm_attr))
        X = model.get_embedding()
    else:
        model.fit(graph)
        X = model.get_embedding()

    print(f"[E_nodes {X.shape}]", end="")

    encoded_nodes = {}
    for i in range(X.shape[0]):
        encoded_nodes[rev_mask[i]] = X[i]
    return encoded_nodes


def F_architecture(graph, mlb=None, mfa=None):
    ### POSITION ENCODING ###
    edges = []
    cluster_feature = {}
    for cluster_idx, cluster in graph.cluster_map.items():
        cluster_feature[cluster_idx] = [len(cluster.nodes) / (1 + len(cluster.edges))]
    for edge in graph.cluster_links:
        cluster_idx_1 = graph.nodes[edge[0]].cluster_idx
        cluster_idx_2 = graph.nodes[edge[1]].cluster_idx
        edges.append([cluster_idx_1, cluster_idx_2])
    P = E_nodes(edges, attr=cluster_feature)

    ### STRUCTURE ENCODING ###
    S = {}
    for cluster_idx in graph.cluster_map.keys():
        edges = graph.cluster_map[cluster_idx].edges
        ## obj = E_nodes(edges)
        if len(edges) > embedding_dim:
            obj = E_nodes(edges)
        else:
            obj = {}
            for idx in graph.cluster_map[cluster_idx].nodes:
                obj[idx] = np.array([0.0] * embedding_dim)  # FIXME: config
        S.update(obj)

    ### NODE ENCODING ###
    N = {}  # FIXME: move to fn_node_encoder?
    vec = []
    for idx, node in graph.nodes.items():
        vec.append(__encode(node.name))
        # vec.append(list(node.name.replace(".weight", "").replace(".bias", "")))
        # vec.append(node.name.split("."))
    vec = mlb.transform(vec)
    vec = mfa.transform(vec)
    vec_final = []
    for i, (idx, node) in enumerate(
        graph.nodes.items()
    ):  # FIXME: better way? [pad len 4]
        _shape4 = nn.ConstantPad1d((0, 4 - len(node.size)), 0.0)(
            torch.tensor(node.size)
        )
        #shape_ab = __shape_score(_shape4.type(torch.FloatTensor), (100, 1, 1, 1))
        #shape_ba = __shape_score(_shape4.type(torch.FloatTensor), (1, 100, 1, 1))
        shape4 = _shape4.type(torch.FloatTensor) / torch.max(1 + _shape4)
        if shape4[0] > shape4[1]:
            rot = 1
        else:
            rot = 0
        _idx_rev = (graph.max_idx - node.idx) / graph.max_idx
        _idx_rev2 = (node.idx) / graph.max_idx
        _level_rev = (graph.max_level - node.level) / graph.max_level
        _level_rev2 = (node.level) / graph.max_level
        _cluster_rev = (graph.max_idx - node.cluster_idx) / graph.max_idx
        _cluster_rev2 = (node.cluster_idx) / graph.max_idx
        _type = 0 if ".bias" in node.name else 1
        # dotcount = node.name.count('.')
        # N[idx] = np.array(
        vec_final.append(np.array(
            [rot]
            + shape4.tolist()
            + [(_idx_rev + _cluster_rev+_level_rev)/3,
               (_idx_rev2+_cluster_rev2+_level_rev2)/3, _type]
        ))
        # vec_final.append(np.array(
        #     # [shape_ab, shape_ba]
        #     [rot]
        #     + shape4.tolist()
        # ))
    from sklearn import preprocessing
    # _pp = preprocessing.QuantileTransformer() # BEST
    # _pp = preprocessing.QuantileTransformer() # 83 / 158
    # _pp = preprocessing.Normalizer(norm='l2') # 77 / 158
    # _pp = preprocessing.Normalizer(norm='l1') # 76 / 158
    # _pp = preprocessing.Normalizer(norm='max') # [78] 79 / 158
    # _pp = preprocessing.PowerTransformer() # 80 / 158
    # _pp = preprocessing.MaxAbsScaler() #XXX 20 77 / 158
    # _pp = preprocessing.RobustScaler() # 78 / 158
    _pp = preprocessing.StandardScaler() #XXX 85 / 158
    # _pp = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal',
    #                                      strategy='quantile') # 75
    vec_final = _pp.fit_transform(vec_final)

    for i, (idx, node) in enumerate(
        graph.nodes.items()
    ):
        # FIXME???????? without vec_final?
        # print(vec_final[i])
        N[idx] = np.array(vec_final[i].tolist() + vec[i].tolist())

    print("(encode_graph ended)")
    return P, S, N


def __q(a, b):
    return np.array(a) + np.array(b)
    # return np.array(a) * np.array(b) # 60 / 158
    # return np.concatenate((a, b), axis=0) # 65 / 158


def __shape_score(s1, s2):
    if len(s1) != len(s2):
        return 0
    score = 1
    for x, y in zip(s1, s2):
        score *= min(x / y, y / x)
    return score


# gen_dataset / `self-learn`
def gen_dataset(graph, P, S, N, EG, prefix=""):
    X, y = [], []

    # FIXME: move to encoder settings? / encoder definition
    for idx, node in graph.nodes.items():
        if node.type != "W":  # FIXME: is it good?
            continue

        cluster_idx = node.cluster_idx
        # FIXME: make it pretty
        # FIXME: encoder score for [N]

        # === CASE 1: [self to self] (q_src, q_dst) -> 1
        for _ in range(CONFIG.samples_per_tensor):
            # FIXME: move to `augmentation`
            p_src = np.array(P[cluster_idx])
            r = np.random.uniform(low=-0.05, high=0.05, size=p_src.shape)
            p_src += r
            s_src = np.array(S[idx])
            r = np.random.uniform(low=-0.05, high=0.05, size=s_src.shape)
            s_src += r
            q_src = p_src.tolist() + s_src.tolist() + list(N[idx]) + \
                EG[f"{prefix}_{idx}"]["in-tree"].tolist()
            X.append(__q(q_src, q_src))
            # FIXME: verify 0.05, 0.05? maybe add as std/var
            y.append(1 + np.random.uniform(low=-0.05, high=0.05))

        q_src = list(P[cluster_idx]) + list(S[idx]) + list(N[idx]) + \
            EG[f"{prefix}_{idx}"]["in-tree"].tolist()

        X.append(__q(q_src, q_src))
        y.append(1)

        def __get_node(cluster_idx=None, type=None):
            r_idx = None
            if cluster_idx is not None:
                nodes = list(graph.cluster_map[cluster_idx].nodes)
            else:
                nodes = list(graph.nodes.keys())
            for _ in range(len(N)):
                r_idx = random.choice(nodes)
                if graph.nodes[r_idx].type == type or not type:
                    break
            return r_idx

        # === CASE 2: same cluster, W
        for _ in range(CONFIG.samples_per_tensor):
            r_idx = __get_node(cluster_idx=cluster_idx, type="W")
            r_cluster_idx = cluster_idx
            if idx == r_idx:
                continue

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx]) + \
                EG[f"{prefix}_{r_idx}"]["in-tree"].tolist()

            N_bonus = 0
            N_dist = np.linalg.norm(N[idx] - N[r_idx])

            if N_dist <= 1:
                N_bonus = (1 - N_dist) / 4

            X.append(__q(q_src, q_dst))
            y.append(
                N_bonus
                + 0.25
                + 0.5 * __shape_score(graph.nodes[idx].size, graph.nodes[r_idx].size)
            )

        # === CASE 3: other cluster, W
        for _ in range(CONFIG.samples_per_tensor):
            r_idx = __get_node(cluster_idx=None, type="W")
            r_cluster_idx = graph.nodes[r_idx].cluster_idx
            if r_cluster_idx == cluster_idx:
                continue
            if idx == r_idx:
                continue

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx]) + \
                EG[f"{prefix}_{r_idx}"]["in-tree"].tolist()

            N_bonus = 0
            N_dist = np.linalg.norm(N[idx] - N[r_idx])

            if N_dist <= 1:
                N_bonus = (1 - N_dist) / 4

            S_bonus = 0
            S_dist = np.linalg.norm(S[idx] - S[r_idx])

            if S_dist <= 1:
                S_bonus = (1 - S_dist) / 4

            X.append(__q(q_src, q_dst))
            y.append(
                N_bonus / 2
                + S_bonus / 2
                + 0.25 * __shape_score(graph.nodes[idx].size, graph.nodes[r_idx].size)
            )

        # === CASE 4: ?, F
        # for _ in range(CONFIG.samples_per_tensor):
        #     r_idx = __get_node(cluster_idx=None, type="F")
        #     r_cluster_idx = graph.nodes[r_idx].cluster_idx
        #     if idx == r_idx:
        #         continue

        #     q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx])

        #     X.append(__q(q_src, q_dst))
        #     y.append(0)

    print("DATASET", np.array(X).shape)#len(y))

    return X, y

# _vec = list(x.replace(".weight", "").replace(".bias", ""))
# # print(_vec)
# _lvl = [s for s in _vec if s.isdigit()]
# _lvl = "".join(_lvl)
# _vec = list(set(_vec))
# if _lvl:
#     _vec.append(_lvl)

def __encode(x):
    x = x.replace(".weight", "").replace(".bias", "")
    x = x.replace("blocks", "")
    if "Backward" in x:
        x = ""
    # print(x)
    _vec = list(x) # + [x]
    # minl, maxl = 1, 2
    # t = x
    # _vec = [t[i:i+j] for i in range(len(t)-minl) for j in range(minl,maxl+1)]
    # print(_vec)
    _lvl = [s for s in _vec if s.isdigit()]
    _lvl = "".join(_lvl)
    _vec = list(set(_vec))
    if _lvl:
        _vec.append(_lvl)
        # for i in range(2, len(_lvl)+1):
        #     _vec.append(_lvl[0:i])
    return _vec

def transfer(model_src, model_dst=None, teacher=None, debug=False):
    # FIXME: replace str to model if needed
    if model_src and model_dst:
        # API: v2
        pass
    elif not model_dst and teacher:
        # API: v1
        model_src, model_dst = teacher, model_src
    else:
        raise Exception("where is teacher?! is this a joke?")

    graph_src = get_graph(model_src)
    graph_dst = get_graph(model_dst)

    # src_ids_to_layers_mapping = get_idx_to_layers_mapping(model_src,
    #                                                           graph_src)
    # dst_ids_to_layers_mapping = get_idx_to_layers_mapping(model_dst,
    #                                                           graph_dst)

    if debug:
        show_graph(graph_src, ver=3, path="__tli_src")
        show_graph(graph_dst, ver=3, path="__tli_dst")

    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.manifold import Isomap

    mlb = MultiLabelBinarizer()

    vec = []
    # FIXME: mutual
    for idx, node in graph_dst.nodes.items():
        # if node.type != "W":
        #    continue
        vec.append(__encode(node.name))
    # for idx, node in graph_src.nodes.items():
    #     vec.append(__encode(node.name))
    #     vec.append(node.name.split("."))
    mlb.fit(vec) # FIXME: 50
    _l1 = len(graph_dst.nodes.keys())
    _l2 = len(graph_dst.cluster_map.keys())
    # print(_l2, _l1)
    mfa = Isomap(n_components=30, n_neighbors=50, p=3) # 30 best
    _vec = mlb.transform(vec)
    mfa.fit(_vec)

    P_src, S_src, N_src = F_architecture(graph_src, mlb=mlb, mfa=mfa)
    P_dst, S_dst, N_dst = F_architecture(graph_dst, mlb=mlb, mfa=mfa)

    from pprint import pprint
    split_map = {}
    for cluster_idx in graph_src.cluster_map.keys():
        _split_map = split_cluster_level(graph_src, cluster_idx)
        for key in _split_map:
            split_map[f"src_{key}"] = _split_map[key]
    print("(graph_src ended)")
    for cluster_idx in graph_dst.cluster_map.keys():
        _split_map = split_cluster_level(graph_dst, cluster_idx)
        for key in _split_map:
            split_map[f"dst_{key}"] = _split_map[key]
    print("(graph_dst ended)")
    EG = encode_graph(split_map)
    # for key in EG:
    #     print("-------->", key, EG[key]["in-tree"].shape)

    X1, y1 = gen_dataset(graph_src, P_src, S_src, N_src, EG, prefix="src")
    X2, y2 = gen_dataset(graph_dst, P_dst, S_dst, N_dst, EG, prefix="dst")
    X = X1 + X2
    y = y1 + y2

    print("DATASET FULL", np.array(X).shape)
    # for x in range(len(X)):
    #     print(np.array(X[x]).shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG.test_size, random_state=42
    )

    ### AUTOENCODER ###

    # https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html#sklearn.semi_supervised.SelfTrainingClassifier

    model = copy(CONFIG.autoencoder)
    ### model.fit(X1, y1)
    ### model.fit(X2, y2)
    model.fit(X_train, y_train)

    #########################

    y_hat = model.predict(X_test)
    loss = mean_squared_error(y_test, y_hat)

    #################################################################
    ## FIXME: bipartie_matching between top-k #######################
    ## FIXME: match by clusters --> if best in cluster / eliminate ##
    ## FIXME: try connection? # FIXME: elimination? greedy {top 3} ##
    #################################################################

    ### MATCHING ###

    # FIXME: move to [fn_matcher, fn_scorer]

    def __norm_weights(graph):
        arr, imap, i = [], {}, 0
        for _, (idx, node) in enumerate(graph.nodes.items()):
            if node.type != "W":
                continue
            arr.append(idx)
            imap[idx] = i
            i += 1
        return arr, imap

    src_arr, src_map = __norm_weights(graph_src)
    dst_arr, dst_map = __norm_weights(graph_dst)

    remap = {}

    n, m = len(src_arr), len(dst_arr)
    scores = np.zeros((n, m))

    # classes = [
    #         nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    #         nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    #         nn.Linear
    # ]

    for dst_j, idx_dst in enumerate(dst_arr):
        node_dst = graph_dst.nodes[idx_dst]
        dst_type = node_dst.name.split(".")[-1]

        q_dst = (
            list(P_dst[node_dst.cluster_idx])
            + list(S_dst[idx_dst])
            + list(N_dst[idx_dst])
            + list(EG[f"dst_{idx_dst}"]["in-tree"].tolist())
        )

        q_arr = []
        for src_i, idx_src in enumerate(src_arr):
            node_src = graph_src.nodes[idx_src]
            src_type = node_src.name.split(".")[-1]

            q_src = (
                list(P_src[node_src.cluster_idx])
                + list(S_src[idx_src])
                + list(N_src[idx_src])
                + list(EG[f"src_{idx_src}"]["in-tree"].tolist())
            )
            q_arr.append(__q(q_src, q_dst))
            scores[src_i, dst_j] = __shape_score(node_dst.size, node_src.size)

            # src_layer = src_ids_to_layers_mapping[idx_src]
            # dst_layer = dst_ids_to_layers_mapping[idx_dst]

            # not_same_class = True
            # for classname in classes:
            #     if isinstance(src_layer, classname) and \
            #         isinstance(dst_layer, classname):
            #             not_same_class = False
            #             break

            if dst_type != src_type:  # or not_same_class:
                scores[src_i, dst_j] = 0

        y_hat = model.predict(q_arr)
        scores[:, dst_j] *= y_hat

    ##############################################

    # for size in np.arange(0.10, 0.50, 0.10):
    #     window_size = size
    #     for _dst_j, idx_dst in enumerate(dst_arr[::-1]):
    #         dst_j = m - _dst_j - 1
    #         ith = dst_j / m
    #         shift = max(int(ith*n - window_size*n), 0)
    #         i = np.argmax(scores[shift:shift+int(window_size*n), dst_j])+shift
    #         if idx_dst not in remap and scores[i, dst_j] > 1 - size:
    #             remap[idx_dst] = src_arr[i]

    beta = 0.5
    smap = copy(scores)
    for _ in range(n*m):
        i, j = np.unravel_index(smap.argmax(), smap.shape)
        smap[i, :] *= beta
        # smap[:, j] *= 0.9 # FIXME
        if dst_arr[j] not in remap:
            smap[:, j] = 0
            remap[dst_arr[j]] = src_arr[i]

    window_size = 0.25
    for _dst_j, idx_dst in enumerate(dst_arr[::-1]):
        dst_j = m - _dst_j - 1
        ith = dst_j / m
        shift = max(int(ith*n - window_size*n), 0)
        i = np.argmax(smap[shift:, dst_j])+shift
        if idx_dst not in remap:
            remap[idx_dst] = src_arr[i]

    ##############################################

    seen = set()
    all_scores = []
    error_n, error_sum = 0, 0
    for j, idx_dst in enumerate(dst_arr):
        node_dst = graph_dst.nodes[idx_dst]

        idx_src = remap[idx_dst]
        score = scores[src_map[idx_src], j]  # src_i, dst_i
        all_scores.append(score)

        name_src = graph_src.nodes[idx_src].name
        name_dst = node_dst.name
        color_code = "\x1b[1;37;40m"
        if name_src != name_dst:
            error_sum += 1
            color_code = "\x1b[1;31;40m"

        color_end = "\x1b[0m"
        print(
            f"src= {idx_src:3} | dst= {idx_dst:3} | "
            + f"S= {round(score, 2):4} | {color_code}{name_src:30}{color_end} / "
            + f"{name_dst:10}"
        )

        seen.add(idx_src)
        error_n += 1

    sim = max(0, min(1, np.mean(all_scores)))

    print("=== MATCH =================")
    print(f" LOSS --> {loss}")
    n = len(graph_src.nodes.keys())
    print(f"  SIM --> \x1b[0;34;40m{round(sim, 4)}\x1b[0m")
    print(f" SEEN --> {len(seen):5} / {n:5} | {round(len(seen)/n,2)}")
    print(f"ERROR --> {error_sum:5} / {error_n:5} | {round(error_sum/error_n,2)}")
    print("===========================")

    #############################################

    # FIXME: dwa razy odpalone?
    # FIXME: choose bigger model to smaller? --> argmax [matrix]
    # FIXME: wes argmax dla wiekszego modelu?
    # FIXME: [(maximum cover, max flow, biparte)]

    if debug:
        # FIXME: do pracy dodac rysunek z sieci typu "debug"
        show_remap(graph_src, graph_dst, remap, path="__tli_remap")

    # p_src_ref = {}
    # for name, param in model_src.named_parameters():
    #     p_src_ref[name] = param
    # p_dst_ref = {}
    # for name, param in model_dst.named_parameters():
    #     p_dst_ref[name] = param

    # with torch.no_grad():
    #     for idx_dst, idx_src in remap.items():
    #         node_src = graph_src.nodes[idx_src]
    #         node_dst = graph_dst.nodes[idx_dst]
    #         p_src = p_src_ref[node_src.name]
    #         p_dst = p_dst_ref[node_dst.name]
    #         fn_inject(p_src, p_dst)

    return sim, remap, graph_src, graph_dst


################################################################################
# Trace Graph
################################################################################


class Node:
    def __init__(self):
        self.idx = 0
        self.var = None
        self.type = None
        self.size = ()
        self.level = 1
        self.cluster_idx = 1


class Graph:
    def __init__(self):
        self.nodes = None
        self.edges = None

        self.cluster_map = None
        self.cluster_links = None

        self.max_level = None
        self.max_idx = None


class Cluster:
    def __init__(self):
        self.cluster_idx = 0
        self.nodes = []
        self.edges = []


def make_graph(var, params=None) -> Graph:
    graph = Graph()  # FIXME: move to CONFIG
    mod_op = ["AddBackward0", "MulBackward0", "CatBackward"]

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    def __get_type(var):
        node = Node()
        node.var = var
        if hasattr(var, "variable"):
            u = var.variable
            node_name = param_map[id(u)]
            size = list(u.size())
            node.name = node_name
            node.size = size
            node.type = "W"
        else:
            node_name = str(type(var).__name__)
            if node_name in mod_op:
                node.type = "OP"
            else:
                node.type = "F"
            node.name = node_name
        return node

    normal_id_map = {}
    normal_id_iter = [0]

    def __normal_id(var):
        __pointer_idx = id(var)
        if __pointer_idx in normal_id_map:
            return normal_id_map[__pointer_idx]
        else:
            normal_id_map[__pointer_idx] = normal_id_iter[0]
            normal_id_iter[0] += 1
            return normal_id_iter[0] - 1

    def __bfs(graph, degree=2):
        nodes = {}
        edges = {}
        _rev_edges = {}
        _level_map = {}
        _mod_op_map = {}
        visited, queue = set(), collections.deque([graph])
        while queue:
            var = queue.popleft()
            idx_root = __normal_id(var)
            if idx_root not in _level_map:
                _level_map[idx_root] = 1
            if idx_root not in _mod_op_map:
                _mod_op_map[idx_root] = idx_root
            if idx_root not in nodes:  # FIXME: for root? yes?
                nodes[idx_root] = __get_type(var)
                nodes[idx_root].cluster_idx = idx_root
                nodes[idx_root].type = "OP"
            if idx_root not in edges:
                edges[idx_root] = []
            if idx_root not in _rev_edges:
                _rev_edges[idx_root] = []
            for _u in var.next_functions:
                u = _u[0]
                idx = __normal_id(u)
                if not u:
                    continue
                edges[idx_root].append(idx)
                if idx not in _rev_edges:
                    _rev_edges[idx] = []
                _rev_edges[idx].append(idx_root)
                if u not in visited:
                    _level_map[idx] = _level_map[idx_root] + 1
                    node = __get_type(u)
                    node.idx = idx
                    if node.type == "OP":
                        _mod_op_map[idx] = idx_root
                    else:
                        _mod_op_map[idx] = _mod_op_map[idx_root]
                    node.level = _level_map[idx]
                    node.cluster_idx = _mod_op_map[idx]
                    nodes[idx] = node
                    # print(f"--> {node.name:30} | {_level_map[idx]:10} " + \
                    #      f">> {_mod_op_map[idx]:10}")
                    visited.add(u)
                    queue.append(u)
        ### === split by degree
        ## FIXME: add min. [branch depth?]
        ## FIXME: next tour (remove "dummy nodes" / [is_op->is_op])
        if degree:
            visited, queue = set(), collections.deque([graph])
            for idx_root in _rev_edges:
                # print(f"----> root {nodes[idx_root].name:50} {len(_rev_edges[idx_root])}")
                if len(_rev_edges[idx_root]) >= degree:
                    # print("\t[MATCH]")
                    nodes[idx_root].type = "OP"
            while queue:
                var = queue.popleft()
                idx_root = __normal_id(var)
                for _u in var.next_functions:
                    u = _u[0]
                    idx = __normal_id(u)
                    if not u:
                        continue
                    if u not in visited:
                        node = nodes[idx]
                        if node.type == "OP":
                            _mod_op_map[idx] = idx_root
                        else:
                            _mod_op_map[idx] = _mod_op_map[idx_root]
                        node.cluster_idx = _mod_op_map[idx]
                        nodes[idx] = node
                        visited.add(u)
                        queue.append(u)
        max_level = 0
        for _, node_level in _level_map.items():
            max_level = max(max_level, node_level)
        return nodes, edges, max_level

    if isinstance(var, tuple):
        raise Exception("Lord Dark Tensor: have not implemented that feature")
        sys.exit(1)
        for v in var:
            __bfs(v.grad_fn)
    else:
        # FIXME: option to choose method? (degree=None)
        # FIXME: add to config
        nodes, edges, max_level = __bfs(var.grad_fn)  # , degree=None)

    graph.nodes = nodes
    graph.edges = edges

    # make clusters
    graph.cluster_map, graph.cluster_links = make_clusters(graph)
    if len(graph.cluster_map.keys()) <= 1:
        graph.cluster_links.append([0, 0])

    # graph meta
    graph.max_level = max_level
    graph.max_idx = normal_id_iter[0]

    return graph


def make_clusters(graph):
    cluster_map = {}
    cluster_links = []
    for idx, node in graph.nodes.items():
        if node.cluster_idx not in cluster_map:
            # print(f"creating cluster {node.cluster_idx}")
            cluster_map[node.cluster_idx] = Cluster()
        cluster_map[node.cluster_idx].nodes.append(idx)
    for idx_root, edges in graph.edges.items():
        node_root = graph.nodes[idx_root]
        for idx in edges:
            if graph.nodes[idx].type == "OP":
                cluster_links.append([idx, idx_root])
                continue
            cluster_map[node_root.cluster_idx].edges.append([idx, idx_root])
    return cluster_map, cluster_links


def get_graph(model, input=None):
    # FIXME: (automatic) find `input` size (just arr?) / (32, 1, 31, 31)
    graph = None
    input_shape = [input] if input else [(3, 32, 32), (1, 31, 31), (3, 224, 224)]
    for _input_shape in input_shape:
        x = torch.randn(32, *_input_shape)
        try:
            x = x.to(device) # FIXME: more pretty?
            model = model.to(device)
            graph = make_graph(model(x), params=dict(model.named_parameters()))
            break
        except Exception as err:
            print("ERROR", err)
            continue
    if not graph:
        raise Exception("something really wrong!")
    return graph


def get_idx_to_layers_mapping(model: nn.Module, graph: Graph) -> Dict[int, nn.Module]:
    names_to_layers_mapping = {}

    def dfs(model: nn.Module, name_prefix: List[str]):
        for child_name, child in model.named_children():
            dfs(child, name_prefix + [child_name])
        names_to_layers_mapping[".".join(name_prefix)] = model

    dfs(model, [])

    ids_to_layers_mapping = {}
    for node in graph.nodes.values():
        if node.type == "W":
            node_name = node.name.replace(".weight", "").replace(".bias", "")
            layer = names_to_layers_mapping[node_name]
            ids_to_layers_mapping[node.idx] = layer

    return ids_to_layers_mapping


################################################################################
# Visualization
################################################################################


def make_dot(graph, ver=0, prefix="", rankdir="TB"):
    graph_idx = id(graph)

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
        # rank="same"
    )

    graph_attr = dict(
        rank="same",
        # splines="true",
        rankdir=rankdir,  # rankdir,
        # ratio="compress",
        # overlay="compress",
        # quadtree="true",
        # overlap="prism",
        # overlap_scaling="0.01"
    )

    print(f"graph_idx={graph_idx}")
    graph_name = f"cluster_{graph_idx}"  # if rankdir == "TB" else str(graph_idx)
    dot = Digraph(name=graph_name, node_attr=node_attr, graph_attr=graph_attr)

    cluster_map, cluster_links = graph.cluster_map, graph.cluster_links

    def __show_graph_nodes():
        for idx, node in graph.nodes.items():
            _header_name = (
                f"[c = {node.cluster_idx} / "
                + f"l = {node.level} / "
                + f"idx = {node.idx}]\n{node.name}"
            )
            if node.type == "OP":
                dot.node(prefix + str(idx), _header_name, fillcolor="green")
            elif node.type == "W":
                dot.node(
                    prefix + str(idx),
                    _header_name + f"\n{node.size}",
                    fillcolor="lightblue",
                )
            else:
                dot.node(prefix + str(idx), _header_name)

    def __show_graph_edges():
        for idx_root, edges in graph.edges.items():
            for idx in edges:
                dot.edge(prefix + str(idx), prefix + str(idx_root), color="black")

    def __show_clusters():
        for cluster_idx, cluster in cluster_map.items():
            with dot.subgraph(name=f"cluster_{graph_idx}_{cluster_idx}") as c:
                c.attr(style="filled", color="lightgrey")
                for edge in cluster.edges:
                    c.edge(prefix + str(edge[0]), prefix + str(edge[1]), color="black")
                c.attr(label=f"cluster {cluster_idx}")
                if rankdir == "LR":
                    c.attr(rotate="90", rankdir="LR")

    if ver == 0:  # orginalny przelyw
        __show_graph_nodes()
        __show_graph_edges()

    if ver == 1:  # przeplyw pomiedzy clustrami
        cluster_seen = set()

        for idx, node in graph.nodes.items():
            if node.type == "OP" and node.cluster_idx not in cluster_seen:
                nodes_in_cluster = len(cluster_map[node.cluster_idx].nodes)
                name = f"{node.cluster_idx} ({nodes_in_cluster})"
                dot.node(prefix + str(node.cluster_idx), name, fillcolor="orange")
                cluster_seen.add(node.cluster_idx)

        for edge in cluster_links:
            cluster_idx_1 = graph.nodes[edge[0]].cluster_idx
            cluster_idx_2 = graph.nodes[edge[1]].cluster_idx
            dot.edge(
                prefix + str(cluster_idx_1),
                prefix + str(cluster_idx_2),
                color="darkgreen",
                penwidth="3",
            )

    if ver == 2:  # sciezki w clustrach
        __show_clusters()

    if ver == 3:  # pelny przeplyw pomiedzy clustrami
        __show_graph_nodes()

        for edge in cluster_links:
            # FIXME: constraint="false", minlen="2"
            dot.edge(
                prefix + str(edge[0]),
                prefix + str(edge[1]),
                color="darkgreen",
                minlen="3",
                penwidth="3",
            )

        __show_clusters()

    resize_dot(dot)
    dot.engine = "dot"
    return dot


def resize_dot(dot, size_per_element=0.15, min_size=12):
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
    return size


def show_graph(model, ver=0, path="__tli_debug", input=None):
    # FIXME: warning about 'graphviz'
    if not isinstance(model, Graph):
        graph = get_graph(model, input=input)
    else:
        graph = model
    dot = make_dot(graph, ver=ver, prefix="this")
    dot.render(filename=path)
    os.system(f"rm {path}")
    print("saved to file")


def show_remap(g1, g2, remap, path="__tli_debug"):
    # FIXME: colors? for each cluster?
    # FIXME: show as matrix? A: top-down B: left-right
    dot_g1 = make_dot(g1, ver=3, prefix="src", rankdir="TB")
    dot_g2 = make_dot(g2, ver=3, prefix="dst", rankdir="LR")

    graph_attr = dict(rankdir="TB",)
    dot = Digraph(name="root", graph_attr=graph_attr)
    dot_g2.graph_attr.update(rotate="90")
    ###
    ### dot.graph_attr.update(rank="same", ranksep="5", nodesep="2", pad="2")
    ###
    dot_g2.graph_attr.update(compound="True")
    dot_g1.graph_attr.update(compound="True")
    dot.graph_attr.update(compound="True")  # , peripheries="0")
    dot.subgraph(dot_g2)
    dot.subgraph(dot_g1)
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("gist_rainbow")
    colors = cmap(np.linspace(0, 1, len(g1.cluster_map.keys())))
    colors_map = {}  # FIXME: sorted?
    for (cluster_idx, color) in zip(g1.cluster_map.keys(), colors):
        colors_map[cluster_idx] = color
    for idx_dst, idx_src in remap.items():
        color = colors_map[g1.nodes[idx_src].cluster_idx]
        dot.edge(
            "src" + str(idx_src),
            "dst" + str(idx_dst),
            color=to_hex(color),
            # color="red",
            constraint="false",
            penwidth="5",
            weight="5",
        )
    dot.render(filename=path)
    os.system(f"rm {path}")
    print("saved to file")


################################################################################
# Debug
################################################################################

if __name__ == "__main__":
    if False:
        from research_models import get_model_debug, ResNetUNet

        model_debug = get_model_debug(seed=3, channels=3, classes=10)
        model_unet = ResNetUNet(n_class=6)

    if False:  # 8, 11, 9
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("mnasnet_100")

    if False:  # 0, 5, 0, 2
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("efficientnet_lite0")

    if False:  # 47, 53, 49, 45, 47, 45
        model_A = get_model_timm("efficientnet_lite0")
        model_B = get_model_timm("efficientnet_lite1")

    if False:  # 9, 9, 4, 2
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("efficientnet_lite1")

    if False:  # 2, 5, 0, 0
        model_A = get_model_timm("efficientnet_lite0")
        model_B = get_model_timm("efficientnet_lite0")

    if False:  # [5, 15, 4] 5
        model_A = get_model_timm("mixnet_s")
        model_B = get_model_timm("mixnet_s")

    if True:  # [83, 77, 85, 78] 82
        model_A = get_model_timm("mixnet_s")
        model_B = get_model_timm("mixnet_m")

    if False:  # [26, 23, 26] 22, 21
        model_A = get_model_timm("mixnet_m")
        model_B = get_model_timm("mixnet_s")

    if False:  # [81, 74, 73, 71, 69] 68, 70, 64
        model_A = get_model_timm("efficientnet_lite1")
        model_B = get_model_timm("tf_efficientnet_b0_ap")

    if False:  # Q: [66, 26, 24, 31, 25, 24, 29,] 30, 24, 18
        model_A = get_model_timm("tf_efficientnet_b0_ap")
        model_B = get_model_timm("mnasnet_100")

    if False: # Q: [76, 61, 60, 58, 57, 57, 62] 57, 57, 55
        model_A = get_model_timm("mixnet_s")
        model_B = get_model_timm("mnasnet_100")

    if False:  # not comparable
        model_A = get_model_timm("regnetx_002")
        model_B = get_model_timm("efficientnet_lite0")

    # FIXME: automatic report

    transfer(model_A, model_B, debug=True)  # tli sie

    # FIXME: normalize score [0, 1], maybe mean?
    # model_A = get_model_timm("efficientnet_lite0")
    # model_B = get_model_timm("efficientnet_lite1")
    # sim_ab = get_tli_score(model_A, model_B)
    # sim_ba = get_tli_score(model_B, model_A)
    # print(f"sim_ab = {round(sim_ab, 4)} | sim_ba = {round(sim_ba, 4)}")
