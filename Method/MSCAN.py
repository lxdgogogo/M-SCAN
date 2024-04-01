from Utils.cluster_utils import similarity
from copy import deepcopy
from MLGraph.multilayer_graph import MultilayerGraph
from collections import defaultdict
from time import time


def base_community_detection(graph: MultilayerGraph, miu: int, eps: float, threshold: int):
    """
    一个多层图上基础的社区检测算法
    :param graph: 输入图
    :param miu: 至少有几个邻居
    :param eps: 相似度阈值
    :param threshold: 至少出现几层
    :return: 社区
    问题：
    1. 没有优化： 优化方向：存储起来每个边有多少个已经存在的
    2. 以及动态处理
    4. 可以设计一个anytime算法？并行算法？
    优化点1：如果miu不符合规范，直接返回即可。
    """
    # 首先得到每一条边都出现多少层
    no_visited: set[int] = set(deepcopy(graph.nodes_iterator))
    clusters = []
    node_cluster: dict[int: set] = defaultdict(set)  # 表示每个顶点参与多少个cluster
    cores = set()
    start = time()
    # print(get_multi_neighborhood(graph, 23, threshold, eps))
    while len(no_visited) != 0:
        node = no_visited.pop()
        node_neighbors = get_multi_neighborhood(graph, node, threshold, eps)
        if len(node_neighbors) >= miu:
            cluster = set()
            cluster.add(node)
            clusters.append(cluster)
            Q = [node]
            cores.add(node)
            # if node == 2:
            #     print(node, node_neighbors)
            while len(Q) != 0:
                q = Q.pop()
                R = get_multi_neighborhood(graph, q, threshold, eps)
                for x in R:
                    if x not in cluster:
                        cluster.add(x)
                        M = get_multi_neighborhood(graph, x, threshold, eps)
                        # if x == 2:
                        #     print(x, q, M)
                        if len(M) >= miu:
                            cores.add(x)
                            # if x == 2:
                            #     print("base", x, q, M)
                            Q.append(x)
            no_visited -= cluster
    # print("base scan")
    # print(len(cores), sorted(cores))
    # print(len(clusters))
    for cluster_id, cluster in enumerate(clusters):
        # print(len(cluster), sorted(cluster))
        for node in cluster:
            node_cluster[node].add(cluster_id)
    no_cluster = set(deepcopy(graph.nodes_iterator))
    for cluster in clusters:
        no_cluster -= cluster
    hubs = []
    outliers = []
    end1 = time()

    for v in no_cluster:
        if same_cluster(graph, v, node_cluster, threshold, eps):
            hubs.append(v)
        else:
            outliers.append(v)
    end2 = time()
    # print("time:", end1 - start, end2 - end1)
    # print("hubs", len(hubs), hubs)
    # print("outliers", len(outliers), outliers)
    return clusters, hubs


def same_cluster(graph: MultilayerGraph, u: int, node_cluster, threshold: int, eps: float):
    cluster_ids = set()
    neighbors = get_multi_neighborhood(graph, u, threshold, eps)
    # if u == 1:
    #     print(u, neighbors)
    for v in neighbors:
        if len(node_cluster[v]) >= 1:
            if len(cluster_ids) == 0:
                cluster_ids = node_cluster[v]
            else:
                for cluster_id in node_cluster[v]:
                    for id_now in cluster_ids:
                        if cluster_id != id_now:
                            return True
    return False


def get_multi_neighborhood(graph: MultilayerGraph, u: int, threshold: int, eps: float = 0.0):
    # 在寻找时，只需要找到𝜆层密度大于等于k就行，没必要找到所有的
    edge_dict: dict[int, int] = defaultdict(int)
    eps_neighbors: list[int] = [u]
    visited = [False for _ in graph.nodes_iterator]
    for layer in graph.layers_iterator:
        for v in graph.adjacency_list[layer][u]:
            if visited[v]:
                continue
            if eps == 0 or (similarity(graph, u, v, layer)) >= eps:
                edge_dict[v] += 1
                # print(u, v, layer)
                if edge_dict[v] >= threshold:
                    eps_neighbors.append(v)
                    visited[v] = True

    return eps_neighbors


def if_sim(graph: MultilayerGraph, u: int, v: int, threshold: int, eps: float = 0.0):
    layer_num = 0
    for layer in graph.layers_iterator:
        if similarity(graph, u, v, layer) >= eps:
            layer_num += 1
            if layer_num >= threshold:
                return True
    return False

