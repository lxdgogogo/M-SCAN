from collections import defaultdict

from MLGraph.multilayer_graph import MultilayerGraph


def get_modularity(graph: MultilayerGraph, clusters: dict[int, list[int]], d_total=0):
    if d_total == 0:
        for layer in graph.layers_iterator:
            for node in graph.nodes_iterator:
                d_total += len(graph.adjacency_list[layer][node])
    Dint_LC = 0
    D2_LC = 0
    for cluster in clusters.values():
        for layer in graph.layers_iterator:
            d_L_C = 0
            for node in cluster:
                Dint_LC += len(graph.adjacency_list[layer][node] & set(cluster))
                d_L_C += len(graph.adjacency_list[layer][node])
            D2_LC += d_L_C ** 2
            # d_L = graph.edges_layer[layer] ** 2
            # d_L_ext = len(graph.nodes_iterator) * (len(graph.layers_iterator) - 1)
            # for layer_other in graph.layers_iterator:
            #     if layer_other == layer:
            #         continue
            #     pass
            # result += d_L_C_int
    # result += d_L_C_int - gamma * (d_L_C ** 2) / d_L + beta * d_L_ext
    mod = Dint_LC / d_total - D2_LC / (d_total * d_total)
    return mod


def get_max_one_modularity(graph: MultilayerGraph, clusters: dict[int, list[int]]):
    d_total = 0
    for layer in graph.layers_iterator:
        for node in graph.nodes_iterator:
            d_total += len(graph.adjacency_list[layer][node])
    D_in_LC = 0
    D2_LC = 0
    call_out = []
    for cluster in clusters.values():
        for layer in graph.layers_iterator:
            d_L_C = 0
            for node in cluster:
                D_in_LC += len(graph.adjacency_list[layer][node] & set(cluster))
                call_out.append((node, layer))
                d_L_C += len(graph.adjacency_list[layer][node])
            D2_LC += d_L_C ** 2
            # d_L = graph.edges_layer[layer] ** 2
            # d_L_ext = len(graph.nodes_iterator) * (len(graph.layers_iterator) - 1)
            # result += d_L_C_int
    # result += d_L_C_int - gamma * (d_L_C ** 2) / d_L + beta * d_L_ext
    mod = D_in_LC / d_total - D2_LC / (d_total * d_total)
    print(mod)
    return mod


def add_node(graph: MultilayerGraph, node, d_sum: int, clusters: dict[int, list[int]], D_L_C: dict[int, list[int]]):
    # 返回增加一个顶点以后modularity的增量。
    max_add_mod = 0  # 最大的增量
    max_cluster_id = -1
    for cluster_id, cluster in clusters.items():
        d_l_node_sum = 0  # d_l(node)
        d_l_node_sum_2 = 0  # d_l(node) * D_l_C
        d_l_node_in = 0  # d_l_int(node)
        for layer in graph.layers_iterator:
            d_l_node_in += len(graph.adjacency_list[layer][node] & set(cluster))
            d_l_node_sum += len(graph.adjacency_list[layer][node]) * D_L_C[cluster_id][layer]
            d_l_node_sum_2 += len(graph.adjacency_list[layer][node]) ** 2
        add_mod = (2 * d_sum * d_l_node_in - 2 * d_l_node_sum - d_l_node_sum_2) / (d_sum ** 2)
        if add_mod > max_add_mod:
            max_add_mod = add_mod
            max_cluster_id = cluster_id
    return max_add_mod, max_cluster_id


def union_cluster_mod(graph: MultilayerGraph, id_1: int, id_2: int, cluster_1: list[int],
                  cluster_2: list[int], d_sum: int, D_L_C: dict[int, list[int]]):
    d_cross = 0
    d_c_1_c_2 = 0
    for layer in graph.layers_iterator:
        d_c_1_c_2 += D_L_C[id_1][layer] * D_L_C[id_2][layer]
        for node_1 in cluster_1:
            for node_2 in cluster_2:
                d_cross += len(graph.adjacency_list[layer][node_1] & graph.adjacency_list[layer][node_2])
    d_c_1_c_2 = 2 * d_c_1_c_2 / (d_sum ** 2)
    d_cross = 2 * d_cross / d_sum
    change_mod = d_cross - d_c_1_c_2
    return change_mod


def delete_node(graph: MultilayerGraph, d_sum: int, clusters: dict[int, list[int]], D_L_C: dict[int, list[int]]):
    # 返回删除一个顶点以后modularity的减。
    max_delete_mod = 0  # 最大的增量
    max_cluster_id = -1
    max_node = -1
    for cluster_id, cluster in clusters.items():
        d_l_node_sum = 0  # d_l(node)
        d_l_node_sum_2 = 0  # d_l(node) * D_l_C
        d_l_node_in = 0  # d_l_int(node)
        for node in cluster:
            for layer in graph.layers_iterator:
                d_l_node_in += len(graph.adjacency_list[layer][node] & set(cluster))
                d_l_node_sum += len(graph.adjacency_list[layer][node]) * D_L_C[cluster_id][layer]
                d_l_node_sum_2 += len(graph.adjacency_list[layer][node]) ** 2
            delete_mod = (2 * d_l_node_sum + d_l_node_sum_2 - 2 * d_sum * d_l_node_in) / (d_sum ** 2)
            if delete_mod > max_delete_mod:
                max_delete_mod = delete_mod
                max_cluster_id = cluster_id
                max_node = node
    if max_delete_mod > 0:
        clusters[max_cluster_id].remove(max_node)
        for layer in graph.layers_iterator:
            D_L_C[max_cluster_id][layer] -= graph.degree_list[max_node][layer]
        if len(clusters[max_cluster_id]) == 0:
            clusters.pop(max_cluster_id)
    return max_delete_mod, max_node
