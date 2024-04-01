import os
from collections import defaultdict
from copy import deepcopy
from random import random
from time import time
from collections import Counter

from Method.MPSCAN import cluster_non_core
from Utils.DSU import DSU
from Utils.cluster_utils import similarity
from MLGraph.multilayer_graph import MultilayerGraph


def gen_interval_index(graph: MultilayerGraph, write=False):
    """
    加入modularity元素。
    :param write: 是否写文件
    :param graph: 多层图
    :return: 对于每个顶点，对于可能的lambda，建立dict
    """
    node_sim_dict: list[dict[int, list[float]]] = [defaultdict(list) for _ in graph.nodes_iterator]
    stab_list: list[list[dict[float, int]]] = []
    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            for neighbor in graph.adjacency_list[layer][node]:
                sim = similarity(graph, node, neighbor, layer)
                node_sim_dict[node][neighbor].append(sim)
    threshold_max = [0 for _ in graph.nodes_iterator]
    for node in graph.nodes_iterator:
        for neighbor_list in node_sim_dict[node].values():
            neighbor_list.sort(reverse=True)
            threshold_max[node] = max(threshold_max[node], len(neighbor_list))
        stab_list.append([{} for _ in range(threshold_max[node] + 1)])
    for node in graph.nodes_iterator:
        for threshold in range(1, threshold_max[node]+1):
            sim_list = []
            for neighbor_list in node_sim_dict[node].values():
                if len(neighbor_list) >= threshold:
                    sim_list.append(neighbor_list[threshold - 1])
            sim_list.sort()
            res = dict(Counter(sim_list))
            eps_ori = 0
            for eps in sorted(res.keys(), reverse=True):
                eps_num = res[eps]
                eps_ori += eps_num
                stab_list[node][threshold][eps] = eps_ori
    if write:
        path = f'../stab_index/{graph.dataset_path}.txt'
        f = open(path, 'w+')
        for node in graph.nodes_iterator:
            f.write(f'{node}\n')
            for threshold in range(1, threshold_max[node] + 1):
                f.write(f'{threshold}')
                for eps, mu in stab_list[node][threshold].items():
                    f.write(f'{mu}, {round(eps, 3)}')
                f.write('\n')
    return stab_list


def find_vertices(stab_list: list[list[dict[int, list[int]]]], miu: int, eps: float, threshold: int):
    core = set()
    for node in range(len(stab_list)):
        if threshold >= len(stab_list[node]):
            continue
        miu_max = 0
        for eps_stab, miu_stab in stab_list[node][threshold].items():
            if eps <= eps_stab:
                miu_max = miu_stab
        if miu_max >= miu:
            core.add(node)
    return core


def query_by_interval_index(graph: MultilayerGraph, miu: int, eps: float, threshold: int, stab_list):
    if threshold > graph.number_of_layers:
        raise Exception('threshold过大')
    core = find_vertices(stab_list, miu, eps, threshold)
    # print(len(core), sorted(core))
    dsu = DSU(len(graph.nodes_iterator))
    for node in core:
        for neighbor, layer_neighbor in graph.node_sim_dict[node].items():
            if neighbor in core and dsu.find(node) != dsu.find(neighbor) and layer_neighbor >= threshold:
                now_number = 0
                for layer in graph.layers_iterator:
                    if neighbor not in graph.adjacency_list[layer][node]:
                        continue
                    sim = similarity(graph, node, neighbor, layer)
                    if sim >= eps:
                        now_number += 1
                    else:
                        layer_neighbor -= 1
                    if now_number >= threshold or layer_neighbor < threshold:
                        break
                if now_number >= threshold:
                    dsu.union(node, neighbor)
                    break
    core_now = set()
    clusters: dict[int, list[int]] = {}
    end2 = time()
    for node in graph.nodes_iterator:
        if node not in core:
            continue
        root = dsu.find(node)
        if root not in core_now:
            clusters[root] = [root]
            core_now.add(root)
        if root != node:
            clusters[root].append(node)
        core_now.add(node)
    non_core = graph.nodes_iterator - core
    print("non_core")
    cluster_non_core(graph, threshold, eps, clusters, non_core)
    # add_non_core(graph, threshold, eps, clusters, non_core)
    # for cluster_id, cluster in clusters.items():
    #     print(len(cluster), sorted(cluster))
    return clusters


if __name__ == '__main__':
    graph = MultilayerGraph('RM')
    stab_list = gen_interval_index(graph, write=True)
    query_by_interval_index(graph, 5, 0.5, 2, stab_list)
