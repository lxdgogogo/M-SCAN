from Utils.DSU import DSU
from Utils.cluster_utils import similarity
from copy import deepcopy
from MLGraph.multilayer_graph import MultilayerGraph
from collections import defaultdict
from time import time

from Utils.modularity_utils import get_modularity


def optimized_PMSCAN_detection(graph: MultilayerGraph, miu: int, eps: float, threshold: int):
    """
    对basecd的改善
    :param graph: 输入图
    :param miu: 至少有几个邻居
    :param eps: 相似度阈值
    :param threshold: 至少出现几层
    :return: 社区

    """
    # print("improve")
    # 首先得到每一条边都出现多少层
    start = time()
    visited: list[bool] = [False for _ in graph.nodes_iterator]
    sd_list: list[int] = [0 for _ in graph.nodes_iterator]
    ed_list: list[int] = [0 for _ in graph.nodes_iterator]
    for node in graph.nodes_iterator:
        for neighbor, layer_num in graph.node_sim_dict[node].items():
            if layer_num >= threshold:
                ed_list[node] += 1
    clusters = {}
    dsu = DSU(len(graph.nodes_iterator))
    node_cluster: dict[int: set] = defaultdict(set)  # 表示每个顶点参与多少个cluster
    for node in graph.nodes_iterator:
        check_core(graph, node, miu, threshold, eps, sd_list, ed_list, visited)
        if sd_list[node] >= miu:
            cluster_core(graph, threshold, miu, eps, node, sd_list, ed_list, dsu, visited)
    core_now = set()
    for node in graph.nodes_iterator:
        if sd_list[node] < miu:
            continue
        root = dsu.find(node)
        if root not in core_now:
            clusters[root] = [root]
            core_now.add(root)
        if root != node:
            clusters[root].append(node)
        core_now.add(node)
    # print(len(core_now), sorted(core_now))
    # print(clusters)
    clusters_cores = deepcopy(clusters)
    non_cores = deepcopy(graph.nodes_iterator)
    non_cores -= core_now
    cluster_non_core(graph, threshold, eps, clusters, non_cores)
    for cluster_id, cluster in clusters.items():
        # print(len(cluster), sorted(cluster))
        for node in cluster:
            node_cluster[node].add(cluster_id)
    no_cluster = set(deepcopy(graph.nodes_iterator))
    for cluster in clusters.values():
        no_cluster -= set(cluster)
    hubs = []
    outliers = []
    for v in no_cluster:
        if same_cluster(graph, v, node_cluster, threshold, eps, no_cluster):
            hubs.append(v)
        else:
            outliers.append(v)
    # print('core', core_now)
    return clusters, hubs, outliers


def check_core(graph: MultilayerGraph, node: int, miu: int, threshold: int, eps: float, sd_list: list[int],
               ed_list: list[int], visited: list[bool]):
    visited[node] = True
    need_visited = set()
    layer_num = graph.number_of_layers
    if ed_list[node] >= miu > sd_list[node]:
        ed_list[node] = 0
        sd_list[node] = 0
        for neighbor, layer_neighbor in graph.node_sim_dict[node].items():
            if layer_neighbor >= threshold:
                need_visited.add(neighbor)
                ed_list[node] += 1
        if ed_list[node] < miu:
            return
        for neighbor in need_visited:
            now_number = 0
            if graph.degree_max[node] < eps * eps * graph.degree_min[neighbor] or graph.degree_max[neighbor] < eps * eps * graph.degree_min[node]:
                continue
            for layer in graph.layers_iterator:
                if neighbor not in graph.adjacency_list[layer][node] and neighbor != node:
                    continue
                sim = similarity(graph, node, neighbor, layer)
                if sim >= eps:
                    now_number += 1
                if now_number >= threshold:
                    sd_list[node] += 1
                    break
                if now_number + layer_num - layer <= threshold:
                    ed_list[node] -= 1
                    break
            if not visited[neighbor]:
                if now_number >= threshold:
                    sd_list[neighbor] += 1
                else:
                    ed_list[neighbor] -= 1
            if sd_list[node] >= miu or ed_list[node] < miu:
                break


def cluster_core(graph: MultilayerGraph, threshold: int, miu: int, eps: float, node: int, sd_list: list[int],
                 ed_list: list[int], dsu: DSU, visited: list[bool]):
    for v, layer_v in graph.node_sim_dict[node].items():
        if layer_v >= threshold and dsu.find(node) != dsu.find(v) and sd_list[v] >= miu:
            now_number = 0
            for layer in graph.layers_iterator:
                if v not in graph.adjacency_list[layer][node] and v != node:
                    continue
                sim = similarity(graph, node, v, layer)
                if sim >= eps:
                    now_number += 1
                else:
                    layer_v -= 1
                if now_number >= threshold or layer_v < threshold:
                    break
            if not visited[v]:
                if now_number >= threshold:
                    sd_list[v] += 1
                else:
                    ed_list[v] -= 1
            if now_number >= threshold:
                dsu.union(node, v)


def cluster_non_core(graph: MultilayerGraph, threshold: int, eps: float,
                     communities: dict[int, list[int]], non_cores: list[int]):
    for community in communities.values():
        for node in community.copy():
            for v, layer_v in graph.node_sim_dict[node].items():
                if v in non_cores and layer_v >= threshold and v not in community:
                    now_number = 0
                    for layer in graph.layers_iterator:
                        if v not in graph.adjacency_list[layer][node] and v != node:
                            continue
                        sim = similarity(graph, node, v, layer)
                        if sim >= eps:
                            now_number += 1
                        else:
                            layer_v -= 1
                        if now_number >= threshold:
                            community.append(v)
                            break
                        if layer_v < threshold:
                            break


def same_cluster(graph: MultilayerGraph, u: int, node_cluster, threshold: int, eps: float, no_cluster: set[int]):
    neighbors: list[int] = [u]
    # 第一次过滤
    for v, layer_v in graph.node_sim_dict[u].items():
        if v in no_cluster or layer_v < threshold or v == u:
            continue
        layer_v = graph.node_sim_dict[u][v]
        now_number = 0
        for layer in graph.layers_iterator:
            if v not in graph.adjacency_list[layer][u] and v != u:
                continue
            sim = similarity(graph, u, v, layer)
            if sim >= eps:
                now_number += 1.
            else:
                layer_v -= 1
            if now_number >= threshold:
                neighbors.append(v)
                break
            if layer_v < threshold:
                break
    cluster_ids = set()
    for v in neighbors:
        if len(node_cluster[v]) >= 1:
            if len(cluster_ids) == 0:
                cluster_ids = node_cluster[v]
            else:
                for cluster_id in node_cluster[v]:
                    for id_now in cluster_ids:
                        if cluster_id != id_now:
                            print('hub: ', u, cluster_id, id_now)
                            return True
    return False
