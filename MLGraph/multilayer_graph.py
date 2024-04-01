import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from random import random


class MultilayerGraph:
    def __init__(self, dataset_path=None, dynamic=False):
        # ****** instance variables ******
        self.dynamic = dynamic
        # layers
        self.number_of_layers = 0
        self.layers_iterator = set()
        self.layers_map = {}
        # nodes and adjacency list
        self.number_of_nodes = 0
        self.maximum_node = 0
        self.nodes_iterator = set()
        self.edges_layer: list[int] = []
        # 第一个索引是层，第二个索引是节点，第三个为该节点在该层上连接的边
        self.adjacency_list: list[list[set[int]]] = []
        self.edge_truss_number: list[dict[tuple, int]] = []
        self.degree_list: list[list[int]] = []  # 表示每个顶点在每一个层上都有多少个邻居
        self.node_sim: list[list[int]] = []
        self.node_sim_dict: list[dict[int]] = []
        self.degree_max: list[int] = []
        self.degree_min: list[int] = []
        self.add_edges = set()
        # 每一层每一个边对应的truss number

        # if dataset_path has been specified
        if dataset_path is not None:
            # read the graph from the specified path
            self.load_dataset(dataset_path)
            # set the dataset path
            self.dataset_path = dataset_path

    def load_dataset(self, dataset_path):
        dataset_file_url = os.path.abspath(os.path.dirname(os.getcwd())) + '/Datasets/' + dataset_path + '.txt'
        # dataset_file_url = os.getcwd() + '/Datasets/' + dataset_path + '.txt'
        dataset_file = open(dataset_file_url)
        # read the first line of the file
        first_line = dataset_file.readline()
        split_first_line = first_line.split(' ')

        # set the number of layers
        self.number_of_layers = int(split_first_line[0])
        self.layers_iterator = set(range(self.number_of_layers))
        # set the number of nodes
        self.number_of_nodes = int(split_first_line[1])
        self.maximum_node = int(split_first_line[2])
        self.nodes_iterator = set(range(self.maximum_node + 1))
        # create the empty adjacency list
        self.adjacency_list: list[list[set[int]]] = [[set() for _ in self.nodes_iterator] for _ in self.layers_iterator]
        self.degree_list: list[list[int]] = [[0 for _ in self.layers_iterator] for _ in self.nodes_iterator]
        # self.node_sim: list[list[int]] = [[0 for _ in self.nodes_iterator] for _ in self.nodes_iterator]
        self.node_sim_dict: list[dict[int]] = [defaultdict(int) for _ in self.nodes_iterator]
        self.degree_max: list[int] = [0 for _ in self.nodes_iterator]
        self.degree_min: list[int] = [sys.maxsize for _ in self.nodes_iterator]
        self.edges_layer: list[int] = [0 for _ in self.layers_iterator]
        # map and oracle of the layers
        layers_map = {}
        layers_oracle = 0

        # for each line of the file
        for _, line in enumerate(dataset_file):
            # split the line
            split_line = line.split(' ')
            layer = int(split_line[0])
            from_node = int(split_line[1])
            to_node = int(split_line[2])

            # if the layer is not in the map
            if layer not in layers_map:
                # add the mapping of the layer
                layers_map[layer] = layers_oracle
                self.layers_map[layers_oracle] = layer
                # increment the oracle
                layers_oracle += 1

            # add the undirected edge
            if self.dynamic and random() < 0.1:
                self.add_edges.add((from_node, to_node, layers_map[layer]))
            else:
                self.add_edge(from_node, to_node, layers_map[layer])
        for layer in self.layers_iterator:
            for node in self.nodes_iterator:
                self.degree_list[node][layer] = len(self.adjacency_list[layer][node])
        for layer in self.layers_iterator:
            for node in self.nodes_iterator:
                self.node_sim_dict[node][node] += 1
                self.degree_max[node] = max(self.degree_max[node], len(self.adjacency_list[layer][node]))
                self.edges_layer[layer] += len(self.adjacency_list[layer][node])/2
                if len(self.adjacency_list[layer][node]) != 0:
                    self.degree_min[node] = min(self.degree_min[node], len(self.adjacency_list[layer][node]))
                for neighbor in self.adjacency_list[layer][node]:
                    self.node_sim_dict[node][neighbor] += 1
                    self.node_sim_dict[neighbor][node] += 1

    def add_edge(self, from_node, to_node, layer):
        # if the edge is not a self-loop
        if from_node != to_node:
            # add the edge
            self.adjacency_list[layer][from_node].add(to_node)
            self.adjacency_list[layer][to_node].add(from_node)

    def get_nodes(self):
        if self.number_of_nodes == self.maximum_node:
            nodes = set(self.nodes_iterator)
            nodes.remove(0)
            return nodes
        else:
            return set(self.nodes_iterator)

    def get_layer_mapping(self, layer):
        return self.layers_map[layer]

    def remove_edge_one_layer(self, layer: int, edge: tuple):
        u, v = edge
        self.adjacency_list[layer][u].remove(v)
        self.adjacency_list[layer][v].remove(u)

    def remove_node(self, node: int):
        """
        从图中删除一个顶点
        :param node: 删除的顶点
        """
        for layer in self.layers_iterator:
            for neighbor in self.adjacency_list[layer][node]:
                self.remove_edge_one_layer(layer, (node, neighbor))

    def get_degrees_layer_by_layer(self):
        degrees_layer_by_layer: list[dict[int, int]] = [{} for _ in self.layers_iterator]
        for layer in self.layers_iterator:
            for node in self.nodes_iterator:
                degrees_layer_by_layer[layer][node] = len(self.adjacency_list[layer][node])
        return degrees_layer_by_layer

    def get_edges(self, nodes: set[int]):
        edge_dict: dict[tuple[int, int], int] = defaultdict(int)
        for layer in self.layers_iterator:
            for node in nodes:
                for neighbor in self.adjacency_list[layer][node] & nodes:
                    if neighbor > node:
                        edge_dict[(node, neighbor)] = edge_dict[(node, neighbor)] + 1
        return edge_dict

    def get_neighbors(self, node: int, layer: int):
        adjacency = deepcopy(self.adjacency_list[layer][node])
        adjacency.add(node)
        return adjacency

    def get_similarities(self):
        edges: list[set[tuple[int, int]]] = [set() for _ in self.layers_iterator]
        for layer in self.layers_iterator:
            for node in self.nodes_iterator:
                for neighbor in self.adjacency_list[layer][node]:
                    if neighbor > node:
                        edges[layer].add((node, neighbor))
        edge_similarities = [defaultdict(float) for _ in self.layers_iterator]
        for layer in self.layers_iterator:
            for edge in edges[layer]:
                u, v = edge
                v_set = self.adjacency_list[layer][v]
                u_set = self.adjacency_list[layer][u]
                inter = v_set.intersection(u_set)
                if inter == 0:
                    edge_similarities[layer][edge] = 0
                edge_similarities[layer][edge] = (len(inter) + 2) / (math.sqrt((len(v_set) + 1) * (len(u_set) + 1)))
        for layer in self.layers_iterator:
            for node in self.nodes_iterator:
                edge = (node, node)
                edge_similarities[layer][edge] = 1
        return edge_similarities
