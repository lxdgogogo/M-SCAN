import math
from MLGraph.multilayer_graph import MultilayerGraph


def similarity(graph: MultilayerGraph, v: int, u: int, layer: int):
    if v not in graph.adjacency_list[layer][u] and v != u:
        return 0
    v_set = graph.adjacency_list[layer][v]
    u_set = graph.adjacency_list[layer][u]
    # print(v_set, u_set)
    inter = v_set.intersection(u_set)
    # if len(inter) == 0:
    #     return 0
    sim = (len(inter) + 2) / (math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))
    return sim


def neighborhood(graph: MultilayerGraph, v: int, eps, layer: int):
    eps_neighbors: list[int] = []
    for u in graph.adjacency_list[layer][v]:
        if (similarity(graph, u, v, layer)) > eps:
            eps_neighbors.append(u)
    return eps_neighbors
