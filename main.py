from MLGraph.multilayer_graph import MultilayerGraph
from Method.MSCAN import MSCAN_algorithm
from Method.MPSCAN import optimized_PMSCAN_detection

if __name__ == '__main__':
    # terrorist Yeast_2 dblp RM dkpol
    graph = MultilayerGraph('homo')
    # MSCAN_mod, MSCAN_clusters = SCC(graph)
    cluster2 = MSCAN_algorithm(graph, 6, 0.5, 2)
    cluster = optimized_PMSCAN_detection(graph, 6, 0.5, 2)

