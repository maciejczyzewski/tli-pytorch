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


# FIXME: move to class ModelObj
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


################################################################################
# Utils
################################################################################


def apply_reset(model):
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

    random.shuffle(edges)

    for a, b in edges:
        __for_single(a)
        __for_single(b)

    norm_edges = []
    for a, b in edges:
        norm_edges.append([normal_id_map[a], normal_id_map[b]])

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
    from karateclub import FeatherGraph

    model = FeatherGraph(order=1, eval_points=5)
    print("FIT")
    model.fit(graphs)
    print("EMBEDDING")
    X = model.get_embedding()
    print(X.shape)

    return utils_mask_to_map(mask, X)


################################################################################
# TLI
################################################################################


class TLIConfig(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


embedding_dim = 5  # FIXME: was 9, how to find?
CONFIG = TLIConfig(
    {
        # FIXME: move outsite? --> lazy_load?
        "node_embedding_attributed": FeatherNode(
            eval_points=3, order=3, reduction_dimensions=32
        ),
        "node_embedding_neighbourhood": NetMF(
            dimensions=embedding_dim
        ),  # FIXME: use xNetMF
        "autoencoder": MLPRegressor(
            max_iter=50, # FIXME: best 50
            early_stopping=False,
            activation="relu",
            solver="adam",
            hidden_layer_sizes=(100,),
            alpha=0.001,
            verbose=True,
        ),
        "test_size": 0.1,  # FIXME: this is important!
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
        model.fit(graph, norm_attr)
        X = model.get_embedding()
    else:
        model.fit(graph)
        X = model.get_embedding()

    print(f"[E_nodes {X.shape}]", end="")

    encoded_nodes = {}
    for i in range(X.shape[0]):
        encoded_nodes[rev_mask[i]] = X[i]
    return encoded_nodes


def F_architecture(graph):
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
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    vec_names = []
    for idx, node in graph.nodes.items():
        vec = node.name.split(".")
        vec_names.append(vec)
    vec_names = mlb.fit_transform(vec_names)
    # FIXME: only if `weights`
    #print(mlb.classes_)
    #sys.exit()
    for i, (idx, node) in enumerate(graph.nodes.items()):  # FIXME: better way? [pad len 4]
        _shape4 = nn.ConstantPad1d((0, 4 - len(node.size)), 0.0)(
            torch.tensor(node.size)
        )
        shape = _shape4.type(torch.FloatTensor) / torch.max(1 + _shape4)
        _level_rev = (graph.max_level - node.level) / graph.max_level
        _cluster_rev = (graph.max_idx - node.cluster_idx) / graph.max_idx
        _type = 0 if ".bias" in node.name else 1
        # FIXME: illegal "." dot split encoder
        # print(vec_names[i])
        N[idx] = np.array(shape.tolist() + [_cluster_rev, _level_rev, _type] +
                          vec_names[i].tolist())

    print("(encode_graph ended)")
    return P, S, N


def __q(a, b):
    return np.concatenate((a, b), axis=0)

def __shape_score(s1, s2):
    if len(s1) != len(s2):
        return 0
    score = 1
    for x, y in zip(s1, s2):
        score *= min(x / y, y / x)
    return score

# gen_dataset / `self-learn`
def gen_dataset(graph, P, S, N):
    X, y = [], []

    # FIXME: move to encoder settings? / encoder definition
    for idx, node in graph.nodes.items():
        # if node.type != "W":  # FIXME: is it good?
        #    continue

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
            q_src = p_src.tolist() + s_src.tolist() + list(N[idx])
            X.append(__q(q_src, q_src))
            # FIXME: verify 0.05, 0.05? maybe add as std/var
            y.append(1 + np.random.uniform(low=-0.05, high=0.05))

        q_src = list(P[cluster_idx]) + list(S[idx]) + list(N[idx])
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

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx])

            # N_bonus = 0
            # N_dist = np.linalg.norm(N[idx] - N[r_idx])

            # if N_dist <= 0.1:
            #     N_bonus = 0.25

            X.append(__q(q_src, q_dst))
            y.append(
                # N_bonus +
                0.25
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

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx])

            # N_bonus = 0
            # N_dist = np.linalg.norm(N[idx] - N[r_idx])

            # if N_dist <= 0.1:
            #     N_bonus = 0.25

            S_bonus = 0
            S_dist = np.linalg.norm(S[idx] - S[r_idx])

            # if S_dist <= 1:
            #     S_bonus = (1-S_dist)/4

            if S_dist <= 0.001:
                S_bonus = 0.25

            X.append(__q(q_src, q_dst))
            y.append(
                # N_bonus +
                S_bonus
                + 0.25 * __shape_score(graph.nodes[idx].size, graph.nodes[r_idx].size)
            )

        # === CASE 4: ?, F
        for _ in range(CONFIG.samples_per_tensor):
            r_idx = __get_node(cluster_idx=None, type="F")
            r_cluster_idx = graph.nodes[r_idx].cluster_idx
            if idx == r_idx:
                continue

            q_dst = list(P[r_cluster_idx]) + list(S[r_idx]) + list(N[r_idx])

            X.append(__q(q_src, q_dst))
            y.append(0)

    print("DATASET", len(y))

    return X, y


def transfer(model_src, model_dst, debug=False):
    graph_src = get_graph(model_src)
    graph_dst = get_graph(model_dst)

    # show_graph(graph_src, ver=3, path="__tli_src")
    # show_graph(graph_dst, ver=3, path="__tli_dst")

    P_src, S_src, N_src = F_architecture(graph_src)
    P_dst, S_dst, N_dst = F_architecture(graph_dst)

    X1, y1 = gen_dataset(graph_src, P_src, S_src, N_src)
    X2, y2 = gen_dataset(graph_dst, P_dst, S_dst, N_dst)
    X = X1 + X2
    y = y1 + y2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG.test_size, random_state=42
    )

    ### AUTOENCODER ###

    model = CONFIG.autoencoder
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    loss = mean_squared_error(y_test, y_hat)

    #################################################################
    ## FIXME: bipartie_matching between top-k #######################
    ## FIXME: match by clusters --> if best in cluster / eliminate ##
    ## FIXME: try connection? # FIXME: elimination? greedy {top 3} ##
    #################################################################

    ### MATCHING ###

    # FIXME: move to [fn_matcher, fn_scorer]
    # FIXME: get only "W" for scoring? --> re-map array

    def __norm_weights(graph):
        arr, imap = [], {}
        for i, (idx, node) in enumerate(graph.nodes.items()):
            if node.type != "W":
                continue
            arr.append(idx)
            imap[i] = idx
        return arr, imap

    src_arr, src_map = __norm_weights(graph_src)
    dst_arr, dst_map = __norm_weights(graph_dst)

    remap = {}

    n, m = len(src_arr), len(dst_arr)
    scores = np.zeros((n, m))

    for dst_j, idx_dst in enumerate(dst_arr):
        node_dst = graph_dst.nodes[idx_dst]
        # dst_type = node_dst.name.split(".")[-1]

        q_dst = (
            list(P_dst[node_dst.cluster_idx])
            + list(S_dst[idx_dst])
            + list(N_dst[idx_dst])
        )

        q_arr = []
        for src_i, idx_src in enumerate(src_arr):
            node_src = graph_src.nodes[idx_src]

            q_src = (
                list(P_src[node_src.cluster_idx])
                + list(S_src[idx_src])
                + list(N_src[idx_src])
            )
            q_arr.append(__q(q_src, q_dst))
            scores[src_i, dst_j] = \
                __shape_score(node_dst.size, node_src.size)

        y_hat = model.predict(q_arr)
        scores[:, dst_j] *= y_hat


    ##############################################

    for dst_j, idx_dst in enumerate(dst_arr):
        i = np.argmax(scores[:, dst_j])
        idx_src = src_arr[i]
        remap[idx_dst] = idx_src

    ##############################################

    seen = set()
    error_n, error_sum = 0, 0

    for j, idx_dst in enumerate(dst_arr):
        node_dst = graph_dst.nodes[idx_dst]

        idx_src = remap[idx_dst]
        score = 0 # scores[src_i[idx_src], dst_i[idx_]]

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

    print("=== MATCH =================")
    print(f" LOSS --> {loss}")
    n = len(graph_src.nodes.keys())
    print(f" SEEN --> {len(seen):5}/{n:5} | {round(len(seen)/n,2)}")
    print(f"ERROR --> {error_sum:5}/{error_n:5} | {round(error_sum/error_n,2)}")
    print("===========================")

    #############################################

    # FIXME: dwa razy odpalone?
    # FIXME: choose bigger model to smaller? --> argmax [matrix]
    # FIXME: wes argmax dla wiekszego modelu?
    # FIXME: [(maximum cover, max flow, biparte)]

    show_remap(graph_src, graph_dst, remap, path="__tli_remap")

    return remap

def transfer_fna():
    # FIXME: udoskonalony 'levi/fna++'?
    pass

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
        nodes, edges, max_level = __bfs(var.grad_fn)#, degree=None)

    graph.nodes = nodes
    graph.edges = edges

    # make clusters
    graph.cluster_map, graph.cluster_links = make_clusters(graph)

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
    input_shape = input if input else (3, 32, 32)
    x = torch.randn(32, *input_shape)
    graph = make_graph(model(x), params=dict(model.named_parameters()))
    return graph


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
    dot.graph_attr.update(compound="True") #, peripheries="0")
    dot.subgraph(dot_g2)
    dot.subgraph(dot_g1)
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, len(g1.cluster_map.keys())))
    colors_map = {} # FIXME: sorted?
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

    # model_A = get_model_timm("regnetx_002")
    # model_B = get_model_timm("efficientnet_lite0")

    model_A = get_model_timm("mixnet_s")
    model_B = get_model_timm("mixnet_m")

    # lite0->lite1 | 52/194
    # lite1->lite0 | 27/149
    # model_A = get_model_timm("efficientnet_lite0")
    # model_B = get_model_timm("efficientnet_lite1")

    transfer(model_A, model_B, debug=True)  # tli sie
