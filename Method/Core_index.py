import heapq
import sys
from collections import defaultdict
sys.path.append('..')

from MLGraph.multilayer_graph import MultilayerGraph
from Method.MSCAN import if_sim
from Utils.cluster_utils import similarity


def gen_core_index(graph: MultilayerGraph, write=False):
    # 首先得到每两个顶点都在哪些层上相邻
    node_node_sims: list[dict[int, list[int]]] = [defaultdict(list) for _ in graph.nodes_iterator]

    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            for neighbor in graph.adjacency_list[layer][node]:
                node_node_sims[node][neighbor].append(similarity(graph, node, neighbor, layer))

    for node in graph.nodes_iterator:
        node_node_sims[node][node] = [1 for _ in graph.layers_iterator]
        for sim_list in node_node_sims[node].values():
            sim_list.sort(reverse=True)
    max_threshold = graph.number_of_layers
    NO: list[list[list[int]]] = [[[] for _ in graph.nodes_iterator] for _ in range(graph.number_of_layers + 1)]
    for threshold in range(1, graph.number_of_layers + 1):  # 表示threshold大小
        flag = False
        for u in graph.nodes_iterator:
            for v, v_sim_list in node_node_sims[u].items():
                if len(v_sim_list) < threshold:
                    continue
                NO[threshold][u].append(v)
            if len(NO[threshold][u]) != 0:
                flag = True
            NO[threshold][u].sort(key=lambda x: node_node_sims[u][x][threshold - 1], reverse=True)
        if not flag:
            max_threshold = threshold
            break

    for threshold in range(graph.number_of_layers,  max_threshold, -1):
        del NO[threshold]

    CO: list[list[list[int]]] = [[] for _ in range(graph.number_of_layers + 1)]
    for threshold in range(1, graph.number_of_layers + 1):
        miu = 1
        CO[threshold].append([])
        CO[threshold].append([])
        flag = False
        while True:
            miu += 1
            for u in graph.nodes_iterator:
                if len(NO[threshold][u]) >= miu:
                    if len(CO[threshold]) <= miu:
                        CO[threshold].append([])
                    CO[threshold][miu].append(u)
            if len(CO[threshold]) <= miu:
                break
            flag = True
            mcs = {}
            for node in CO[threshold][miu]:
                node_sim = []
                for sim_list in node_node_sims[node].values():
                    if len(sim_list) >= threshold:
                        node_sim.append(sim_list[threshold - 1])
                mcs[node] = heapq.nlargest(miu, node_sim)[-1]
            CO[threshold][miu].sort(key=lambda x: mcs[x], reverse=True)
        if not flag:
            break
    if write:
        f = open('../base_index/%s_NO.txt' % (graph.dataset_path), 'w+')
        for threshold in range(1, max_threshold + 1):
            for node in graph.nodes_iterator:
                if len(NO[threshold][node]) > 0:
                    f.write('%d: %d: %s\n' % (threshold, node, str(NO[threshold][node])))
        f.close()
        f = open('../base_index/%s_CO.txt' % (graph.dataset_path), 'w+')
        for threshold in range(1, max_threshold + 1):
            for miu in range(2, len(CO[threshold])):
                if len(CO[threshold][miu]) > 0:
                    f.write('%d: %d: %s\n' % (threshold, miu, str(CO[threshold][miu])))
        f.close()
    return NO, CO


def query_by_index(graph: MultilayerGraph, miu: int, eps: float, threshold: int, NO=None, CO=None):
    # if CO is None:
    #     f1 = open('../base_index/%s_CO.txt' % graph.dataset_path, 'r')
    #     co_list = []
    #     for line in f1:
    #         str_line = line.split(';')
    #         node_threshold = int(str_line[0])
    #         node_miu = int(str_line[1])
    #         if node_miu == miu and node_threshold == threshold:
    #             co_list = eval(str_line[2])
    # if NO is None:
    #     NO = [[] for _ in graph.nodes_iterator]
    #     f2 = open('../base_index/%s_NO.txt' % graph.dataset_path, 'r')
    #     for i, line in enumerate(f2):
    #         str_line = line.split('and')
    #         node_threshold = int(str_line[0])
    #         node = int(str_line[1])
    #         if threshold == node_threshold:
    #             NO[node] = eval(str_line[2])
    NO = NO[threshold]
    if len(CO[threshold]) <= miu:
        return []
    co_list = CO[threshold][miu]
    # print('cores', len(co_list), co_list)
    clusters = []
    cores = set()
    visited = set()
    for node in co_list:
        if node in visited:
            continue
        if not if_sim(graph, node, NO[node][miu - 1], threshold, eps):
            break
        cluster = set()
        cluster.add(node)
        clusters.append(cluster)
        Q = [node]
        cores.add(node)
        # if node == 2:
        #     print(node, node_neighbors)
        while len(Q) != 0:
            q = Q.pop()
            R = get_multi_neighborhood_NO(graph, q, threshold, eps, NO[q])
            for x in R:
                if x not in cluster:
                    cluster.add(x)
                    if if_sim(graph, node, NO[node][miu - 1], threshold, eps):
                        cores.add(x)
                        Q.append(x)
        visited.update(cluster)

    # for cluster in clusters:
    #     print(len(cluster), sorted(list(cluster)))
    return clusters


def get_multi_neighborhood_NO(graph: MultilayerGraph, u: int, threshold: int, eps: float, no_list):
    eps_neighbors: list[int] = [u]
    for v in no_list:
        if if_sim(graph, u, v, threshold, eps):
            eps_neighbors.append(v)
        else:
            break
    return eps_neighbors
