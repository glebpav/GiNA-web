import json

from nskit import *
from nskit.draw.circular_graph import CircularGraph
from scripts.graph_3d_builder import adjacency_matrix_to_list
from flask import Response, jsonify


def build_circular_graph(structure):
    try:
        na = NA(structure)
        adj = na.get_adjacency()
        adj_list = adjacency_matrix_to_list(adj)
        graph = CircularGraph()
        pos, r = graph.calculate_circular_coords(len(structure))
        return jsonify(json.dumps({'pos': pos.tolist(), 'R': r, 'adj': adj_list}))
    except Exception as e:
        return Response(
            str(e),
            status=409,
        )
