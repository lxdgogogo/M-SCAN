import os
from collections import defaultdict
from copy import deepcopy
from random import random
from time import time

from Method.MPSCAN import cluster_non_core
from Utils.DSU import DSU
from Utils.cluster_utils import similarity
from MLGraph.multilayer_graph import MultilayerGraph


def gen_bucket_index(graph: MultilayerGraph, bucket_num: int = 10, write=False):
    """
    加入modularity元素。
    :param write: 是否写文件
    :param graph: 多层图
    :param bucket_num: 桶的数量，即将1分成几份，然后做
    :return: 一个index，给定eps和threshold，得到树状的索引
    """
    node_sim_dict: list[dict[int, list[float]]] = [defaultdict(list) for _ in graph.nodes_iterator]
    bucket_list: list[list[dict[int, list[int]]]] = []
    sims = []
    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            node_sim_dict[node][node].append(1)
            for neighbor in graph.adjacency_list[layer][node]:
                sim = similarity(graph, node, neighbor, layer)
                node_sim_dict[node][neighbor].append(sim)
                sims.append(sim)
    sims.sort()
    sim_partition = []
    for i in range(bucket_num):
        sim_index = int(i / bucket_num * len(sims))
        sim_partition.append(sims[sim_index])
    for node in graph.nodes_iterator:
        for neighbor_sim_list in node_sim_dict[node].values():
            neighbor_sim_list.sort(reverse=True)
    # 这里，第一个索引表示桶，第二个索引表示threshold，j=1表示至少一层相邻，一次类推
    d_total = 0
    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            d_total += len(graph.adjacency_list[layer][node])
    # 首先计算，每个顶点在给定eps和threshold情况下，有多少个邻居
    for i, partition in enumerate(sim_partition):
        bucket: list[dict[int, (list[int], float)]] = [{}]
        bucket_list.append(bucket)
        for threshold in range(1, graph.number_of_layers + 1):
            core_index: dict[int, list[int]] = defaultdict(list)
            for node in graph.nodes_iterator:
                node_miu = 0
                for neighbor_sim_list in node_sim_dict[node].values():
                    if len(neighbor_sim_list) >= threshold and neighbor_sim_list[threshold - 1] >= partition:
                        node_miu += 1
                core_index[node_miu].append(node)
            if len(core_index) != 0:
                bucket.append({})
            miu_list = sorted(core_index.keys(), reverse=True)
            for miu in miu_list:
                if miu == 1:
                    continue
                core_add = core_index[miu]
                bucket[threshold][miu] = core_add

    # print(sim_partition)
    if write:
        path = f'../bucket_index/{graph.dataset_path}'
        folder = os.path.exists(path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(path)
        for file in os.listdir(path):
            file_path = f'../bucket_index/{graph.dataset_path}/{file}'
            os.remove(file_path)
        for i, partition in enumerate(sim_partition):
            f = open(f'../bucket_index/{graph.dataset_path}/{partition}.txt', 'w+')
            for threshold in range(1, graph.number_of_layers + 1):
                for miu, cores in bucket_list[i][threshold].items():
                    f.write(f'{threshold}; {miu}; {cores}\n')

    return bucket_list, sim_partition


def find_vertices(bucket: dict[int, list[int]], miu: int):
    core = set()
    for rho, core_list in bucket.items():
        if rho >= miu:
            core |= set(core_list)
    return core


def query_by_bucket_core_index(graph: MultilayerGraph, miu: int, eps: float, threshold: int, bucket_list,
                               eps_list: list[float]):
    if threshold > graph.number_of_layers:
        raise Exception('threshold过大')
    bucket_number = -1
    eps_index = 0.0
    for i, eps_index in enumerate(eps_list):
        if eps < eps_index:
            bucket_number = i
            break

    # core_1中不一定全部是，core中必须全都是
    if bucket_number == 0:
        core_1 = deepcopy(graph.nodes_iterator)
    else:
        core_1 = find_vertices(bucket_list[bucket_number - 1][threshold], miu)
    if bucket_number == -1:  # eps_list中最大的也比eps小
        core = set()
    else:
        core = find_vertices(bucket_list[bucket_number][threshold], miu)
    dsu = DSU(len(graph.nodes_iterator))
    if eps == eps_index:
        need_judge = set()
    elif bucket_number == -1:
        need_judge = find_vertices(bucket_list[- 1][threshold], miu)
    else:
        need_judge = core_1 - core
    # print("need judge", len(need_judge))
    layer_num = graph.number_of_layers

    for node in need_judge:
        cnt = 0
        need_visited = set()
        for neighbor, layer_neighbor in graph.node_sim_dict[node].items():
            if layer_neighbor >= threshold:
                need_visited.add(neighbor)
        max_cnt = len(need_visited)
        if max_cnt < miu:
            continue
        for neighbor in need_visited:
            now_number = 0
            if (graph.degree_max[node] < eps * eps * graph.degree_min[neighbor] or
                    graph.degree_max[neighbor] < eps * eps * graph.degree_min[node]):
                continue
            if cnt >= miu or max_cnt < miu:
                continue
            for layer in graph.layers_iterator:
                if neighbor not in graph.adjacency_list[layer][node] and neighbor != node:
                    continue
                sim = similarity(graph, node, neighbor, layer)
                if sim >= eps:
                    now_number += 1
                if now_number >= threshold:
                    cnt += 1
                    break
                if now_number + layer_num - layer <= threshold:
                    max_cnt -= 1
                    break
        if cnt >= miu:
            core.add(node)

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
    # print(end1 - start, end2 - end1)
    # print('core', len(core), sorted(core))
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
    # print(clusters)
    non_core = graph.nodes_iterator - core
    cluster_non_core(graph, threshold, eps, clusters, non_core)
    # add_non_core(graph, threshold, eps, clusters, non_core)
    # for cluster_id, cluster in clusters.items():
    #     print(len(cluster), sorted(cluster))
    return clusters
