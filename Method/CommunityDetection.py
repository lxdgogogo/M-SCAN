import os
import sys
from collections import defaultdict
from copy import deepcopy
from time import time

from Method.MSCAN import get_multi_neighborhood

sys.path.append('..')


from Method.Bucket_index import gen_bucket_index
from MLGraph.multilayer_graph import MultilayerGraph
from Utils.cluster_utils import similarity
from Utils.modularity_utils import get_modularity, add_node, delete_node, union_cluster_mod


def SCAN_CD(graph: MultilayerGraph):
    """
    使用的索引
    :param bucket_list: 桶索引
    :param graph: 输入的多层图
    :return: 返回一个最大密度的clusters。
    这个算法使用的是作为平均值作为目标。显然，不合适。
    """

    start = time()
    cluster_node = set()
    D_L_C: dict[int, list[int]] = {}
    node_sim_dict: list[dict[int, list[float]]] = [defaultdict(list) for _ in graph.nodes_iterator]
    mod_list: list[list[dict[int, float]]]
    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            node_sim_dict[node][node].append(1)
            for neighbor in graph.adjacency_list[layer][node]:
                sim = similarity(graph, node, neighbor, layer)
                node_sim_dict[node][neighbor].append(sim)
    bucket_index, sim_partition = gen_bucket_index(graph)
    print(sim_partition)
    d_total = 0
    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            d_total += len(graph.adjacency_list[layer][node])
    max_modularity = 0
    max_cores = set()
    max_eps = 0
    max_threshold = 0
    for i, bucket in enumerate(bucket_index):
        eps = sim_partition[i]
        for threshold in range(1, graph.number_of_layers + 1):
            core_index = bucket[threshold]
            core_nodes = set()
            miu_list = sorted(core_index.keys(), reverse=True)
            for miu in miu_list:
                core_add = core_index[miu]
                core_nodes |= set(core_add)
                clusters = get_clusters(eps, threshold, core_nodes, node_sim_dict)
                mod = get_modularity(graph, clusters, d_total)
                if mod > max_modularity:
                    max_cores = core_nodes
                    max_modularity = mod
                    max_eps = eps
                    max_threshold = threshold
                bucket[threshold][miu] = core_add
    max_clusters = add_non_core(graph, max_eps, max_threshold, max_cores)
    for cluster_id, cluster in max_clusters.items():
        cluster_node.update(cluster)
        D_L_C[cluster_id] = [0 for _ in graph.layers_iterator]
        for layer in graph.layers_iterator:
            for node in cluster:
                D_L_C[cluster_id][layer] += graph.degree_list[node][layer]
    no_cluster_node = graph.nodes_iterator - cluster_node
    d_sum = 0
    for node in graph.nodes_iterator:
        d_sum += sum(graph.degree_list[node])
    mod = get_modularity(graph, max_clusters)
    union_cluster(graph, max_clusters, d_sum, D_L_C)
    while len(no_cluster_node) > 0:
        max_cluster_id = -1
        # 对cluster内部的点删除
        delete_mod, max_delete_node = delete_node(graph, d_sum, max_clusters, D_L_C)
        mod += delete_mod
        no_cluster_node.add(max_delete_node)
        # 对cluster外部的点加入
        max_add_mod = 0
        max_add_node = -1
        for node in no_cluster_node:
            add_mod, cluster_id = add_node(graph, node, d_sum, max_clusters, D_L_C)
            if add_mod > max_add_mod:
                max_add_mod = add_mod
                max_cluster_id = cluster_id
                max_add_node = node
        if max_delete_node == -1 and max_add_node == -1:
            # 如果没有内部和外部都没有变化，直接break
            break
        print('delete and add', max_delete_node, max_add_node)
        if max_add_mod > 0:
            max_clusters[max_cluster_id].append(max_add_node)
            no_cluster_node.remove(max_add_node)
            mod += max_add_mod
            for layer in graph.layers_iterator:
                D_L_C[max_cluster_id][layer] += graph.degree_list[max_add_node][layer]
    # for cluster_id, cluster in max_clusters.items():
    #     print(len(cluster), cluster_id, cluster)
    end = time()
    file_url = os.path.abspath(
        os.path.dirname(os.getcwd())) + '/modularity_graph/' + graph.dataset_path + '_improve_index_mod.txt'
    # f = open(file_url, 'w')
    # for cluster in max_clusters.values():
    #     f.write(f'{len(cluster)}, {cluster}\n')
    # f.write('time: %f\n' % (end - start))
    # f.close()
    return mod, max_clusters


def union_cluster(graph: MultilayerGraph, max_clusters: dict[int: list[int]], d_sum: int, D_L_C: dict[int, list[int]]):
    change_sum = 0
    while True:
        max_mod = 0
        union_1 = -1
        union_2 = -1
        for id_1, cluster_1 in max_clusters.items():
            for id_2, cluster_2 in max_clusters.items():
                if id_1 <= id_2:
                    continue
                change_mod = union_cluster_mod(graph, id_1, id_2, cluster_1, cluster_2, d_sum, D_L_C)
                if change_mod > max_mod:
                    union_1 = id_1
                    union_2 = id_2
                    max_mod = change_mod
        if max_mod == 0:
            break
        cluster_add = max_clusters[union_2]
        max_clusters[union_1].extend(cluster_add)
        max_clusters.pop(union_2)
        change_sum += max_mod
        for layer in graph.layers_iterator:
            for node in cluster_add:
                D_L_C[union_1][layer] += graph.degree_list[node][layer]
    return change_sum


def get_clusters(eps: float, threshold: int, core: set[int], sim_dict: list[dict[int, list[float]]]):
    clusters = defaultdict(list)
    no_cluster = deepcopy(core)
    id = -1
    for node in core:
        if node not in no_cluster:
            continue
        id += 1
        clusters[id].append(node)
        expand_vertices = set()
        expand_vertices.add(node)
        no_cluster.remove(node)
        flag = True
        while flag:
            flag = False
            for v in expand_vertices.copy():
                expand_vertices.remove(v)
                for v_expand in no_cluster.copy():
                    v_sim = sim_dict[node][v]
                    if len(v_sim) >= threshold and v_sim[threshold - 1] >= eps:
                        expand_vertices.add(v_expand)
                        clusters[id].append(v_expand)
                        no_cluster.remove(v_expand)
                        flag = True
    return clusters


def add_non_core(graph, eps, threshold, cores):
    core_set = set(cores)
    clusters: dict[int: list[int]] = {}
    no_visited = deepcopy(core_set)
    while len(no_visited) != 0:
        node = no_visited.pop()
        cluster = set()
        cluster.add(node)
        Q = set()
        Q.add(node)
        while len(Q) != 0:
            q = Q.pop()
            R = get_multi_neighborhood(graph, q, threshold, eps)
            # Q必须得是q的邻居，核顶点，还不能被访问过..
            Q |= (set(R) & core_set - cluster)
            no_visited -= cluster
            cluster.update(R)
        clusters[node] = list(cluster)
    return clusters


if __name__ == '__main__':
    # terrorist Yeast_2 DBLP2 dblp RM FAO
    # datasets = ['terrorist', 'Yeast_2', 'DBLP2', 'dblp', 'RM', 'FAO']
    datasets = ['terrorist']
    for dataset in datasets:
        graph = MultilayerGraph(dataset)
        SCAN_CD(graph)
