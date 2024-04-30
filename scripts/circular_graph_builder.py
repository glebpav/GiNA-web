import json

from nskit import *
from nskit.draw.circular_graph import CircularGraph
from scripts.graph_3d_builder import adjacency_matrix_to_list


def build_circular_graph(structure):
    na = NA(structure)
    adj = na.get_adjacency()
    adj_list = adjacency_matrix_to_list(adj)
    graph = CircularGraph()
    pos, r = graph.calculate_circular_coords(len(structure))
    return json.dumps({'pos': pos.tolist(), 'R': r, 'adj': adj_list})
