import math
import sys
import json

import numpy as np
import networkx as nx
from flask import Response, jsonify

from naskit import *


def find_centers(pos, loops):
    return [pos[loop].mean(0) for loop in loops]


def adjacency_matrix_to_list(adj_matrix):
    adj_list = {}
    for i in range(len(adj_matrix)):
        neighbors = []
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                neighbors.append(j)
                # neighbors.append(len(adj_matrix[i]) - j)

        if i - 1 > 0:
            neighbors.append(i - 1)
        if i + 1 < len(adj_matrix):
            neighbors.append(i + 1)

        adj_list[i] = neighbors
    return adj_list


def get_loop_radius(target_len, points_number):
    return target_len / (2 * math.sin(math.pi / points_number))


def get_avg_connection_number(pos: np.ndarray, na: NA, loop_points_idxes):
    return np.mean(np.array([
        np.linalg.norm(pos[idx1] - pos[idx2])
        for (idx1, idx2) in na.pairs
        if idx1 in loop_points_idxes and idx2 in loop_points_idxes
    ]))

def optimize(
        na: NA,
        pos,  # list of initiall coordinates
        time_step=0.0001,  # ~ learning rate
        elastic_inf=1.,  # elastic_force influence
        repulsion_inf=1.,  # repulsion_force influence
        center_inf=1.,  # center_force influence
        knots_ratio=0.5,  # elastic force ratio for knots
        helix_ratio=0.9,  # elastic force ratio for helix (not knots)
        iterations=50,
        add_counters_node=True
):
    """! дальше нужно работать с матрицами, никаких циклов по листам"""
    A = adjacency_matrix_to_list(na.get_adjacency())
    A_origin = adjacency_matrix_to_list(na.get_adjacency())

    if add_counters_node:
        for point_idx in range(9, len(pos), 10):
            pos = np.vstack((pos, pos[point_idx] + np.array([0.01, 0.01, 0.])))
            A[point_idx].append(len(pos) - 1)
            A[len(pos) - 1] = [point_idx]

    loops = [l.nts for l in na.loops]
    loop_points_idxes = []
    for loop in loops:
        loop_points_idxes += loop

    loop_points_idxes = [nt for loop in na.loops for nt in loop.nts]

    l0 = get_avg_connection_number(pos, na, loop_points_idxes)
    # l01 = get_avg_connection_number1(pos=pos, na=na)

    pos /= l0
    l0 = 1

    repulsion_mask = np.ones((len(A), len(A)))
    target_edge_ratios = np.zeros((len(A), len(A)))
    edge_mask = np.zeros((len(A), len(A)))
    marker_mask = np.array([500. if idx > len(A_origin) else 1. for idx in range(len(A))])
    displacement = np.zeros((len(A), 3))
    center_force = np.zeros((len(A), 3))

    diagonal_mask = np.eye(len(A), dtype=bool)
    repulsion_mask[diagonal_mask] = 0
    repulsion_mask[:, len(A_origin):] = 0
    repulsion_mask[len(A_origin):, :] = 2

    for idx1, neibs in A.items():
        for idx2 in neibs:
            if idx1 < len(A_origin) and idx2 < len(A_origin):
                edge_mask[idx1, idx2] = 1

            target_edge_ratios[idx1, idx2] = 1
            target_edge_ratios[idx2, idx1] = 1

    """! про итерирование по хеликсу уже написал"""
    for h in na.knot_helixes:
        for idx, (i, j) in enumerate(h):
            target_edge_ratios[i, j] = helix_ratio
            target_edge_ratios[j, i] = helix_ratio

    for knot in [list(na.helixes[idx]) for idx in na.knots]:
        for idx, (i, j) in enumerate(knot):
            dist = int(math.floor(abs((len(knot) - 1) / 2 - idx)))
            target_edge_ratios[i, j] = math.pow(knots_ratio, dist * 2)
            target_edge_ratios[j, i] = math.pow(knots_ratio, dist * 2)

    for iteration in range(iterations):

        displacement *= 0
        centers = find_centers(pos, loops)
        l0 = get_avg_connection_number(pos, na, loop_points_idxes)

        """! этот цикл приемлем, ок"""
        for center_idx, center in enumerate(centers):
            r0 = get_loop_radius(l0, len(loops[center_idx]))
            delta = pos[loops[center_idx]] - center
            distance = np.linalg.norm(delta, axis=1)
            center_force[loops[center_idx]] = (-2) * (1 - r0 / np.where(distance < 0.001, 0.001, distance))[:, np.newaxis] * delta

        delta = pos.reshape(-1, 1, 3) - pos
        dists = np.linalg.norm(delta, axis=-1)
        dists = np.where(dists < 0.001, 0.001, dists)

        # Calculate elastic forces
        elastic_factors = (-2) * (marker_mask * edge_mask * (1 - (l0 * target_edge_ratios / dists)))[:, :, np.newaxis]
        elastic_forces = (elastic_factors * delta).sum(axis=1)
        # print(f"{np.shape(elastic_forces)=}")

        # Calculate repulsion forces
        repulsion_factors = (repulsion_mask * np.power(dists, -3))[:, :, np.newaxis]
        repulsion_forces = (repulsion_factors * delta).sum(axis=1)
        # print(f"{np.shape(repulsion_forces)=}")

        # Calculate displacements
        displacement = elastic_inf * elastic_forces + repulsion_inf * repulsion_forces
        # displacement = repulsion_inf * repulsion_forces
        # displacement = elastic_inf * elastic_forces
        # print(f"{elastic_forces}")
        # print(f"{displacement=};")

        """print("here1")

        elastic_force = (
                (-2) * (1 - (l0 * target_edge_ratios[:len(A), A[:len(A)]] / dists[:len(A), A[:len(A)]])[:, np.newaxis]) * delta[:len(A), A[:len(A)]]
        ).sum(axis=0)

        print("here2")

        repulsion_force = (
                (repulsion_mask[:len(A)] * np.power(dists[:len(A), :], -3))[:, np.newaxis] * delta[:len(A), :]
        ).sum(axis=0)

        print("here3")

        displacement[:len(A)] = (elastic_inf * elastic_force + repulsion_inf * repulsion_force)

        print("here4")"""


        """for i in range(len(A)):

            elastic_force = (
                    (-2) * (1 - (l0 * target_edge_ratios[i, A[i]] / dists[i, A[i]])[:, np.newaxis]) * delta[i, A[i]]
            ).sum(axis=0)

            repulsion_force = (
                    (repulsion_mask[i] * np.power(dists[i, :], -3))[:, np.newaxis] * delta[i, :]
            ).sum(axis=0)

            displacement[i] = (elastic_inf * elastic_force + repulsion_inf * repulsion_force)
            # displacement[i] = (repulsion_inf * repulsion_force)
            # displacement[i] = (elastic_inf * elastic_force)"""

        displacement += center_inf * center_force
        displacement = displacement * time_step
        """displacement = np.where(
            np.linalg.norm(displacement, axis=0) > l0 / 1/3,
            displacement * (l0 / 2 / np.linalg.norm(displacement, axis=0)),
            displacement
        )"""

        pos += displacement

    return pos, A


def compute_graph(sequence):
    na = NA(sequence)
    adj = na.get_adjacency()

    adj_list = adjacency_matrix_to_list(adj)
    G = nx.from_dict_of_lists(adj_list)

    knots_list = []
    for h in na.knot_helixes:
        for i, j in h:
            knots_list.append(i)
            knots_list.append(j)

    """! число итераций должно подбираться в зависимости от размера структуры или числа связей в ней"""
    pos = nx.spring_layout(G, dim=3, iterations=5000)
    pos = np.array([value for value in pos.values()])

    # print(f"Start optimize for {sequence}")
    """! число итераций должно подбираться в зависимости от размера структуры или числа связей в ней
    ! 2000 это потолок"""
    pos, adj_list = optimize(
        na,
        pos,
        elastic_inf=5,
        repulsion_inf=1.21,
        center_inf=10,
        time_step=0.0001,
        iterations=2000,
        knots_ratio=1.2,
        helix_ratio=1.0,
        add_counters_node=True
    )
    # print(f"------ End for {sequence}")
    return pos.tolist(), adj_list, knots_list


def build_3d_graph(structure):
    try:
        pos, adj_list, knots_list = compute_graph(structure)
        return jsonify(json.dumps({'pos': pos, 'adj': adj_list, 'knots': knots_list}))
    except Exception as e:
        print(str(e))
        return Response(str(e), status=422)
