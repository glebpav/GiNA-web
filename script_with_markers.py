import math
import sys
import json

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from nskit import *


def find_centers(pos, loops):
    centers = []
    for l in loops:
        coords = np.array([0., 0., 0.])
        for idx in l:
            coords += pos[idx]
        centers.append(coords / len(l))
    return centers


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
    l0 = 0
    connection_number = 0

    A = adjacency_matrix_to_list(na.get_adjacency())
    A_origin = adjacency_matrix_to_list(na.get_adjacency())

    if add_counters_node:
        for point_idx in range(9, len(pos), 10):
            pos = np.vstack((pos, pos[point_idx] + np.array([0.01, 0.01, 0.])))
            A[point_idx].append(len(pos) - 1)
            A[len(pos) - 1] = [point_idx]

    print(f"{len(A)=}")
    print(f"{len(A_origin)=}")

    loops = [l.nts for l in na.loops]

    for i in range(len(A)):
        for point_idx in A[i]:
            if i == point_idx:
                continue
            connection_number += 1
            l0 += np.linalg.norm(pos[i] - pos[point_idx])

    l0 /= connection_number
    print(f"{l0=}")

    repulsion_mask = np.ones((len(A), len(A)))
    target_edge_ratios = np.ones((len(A), len(A)))
    displacement = np.zeros((len(A), 3))
    center_force = np.zeros((len(A), 3))
    elastic_force = np.array([0., 0., 0.])
    repulsion_force = np.array([0., 0., 0.])

    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                print("i == j")
                repulsion_mask[i][j] = 0
                continue
            if j > len(A_origin):
                print("j > len(A_origin)")
                repulsion_mask[i][j] = 0
                continue
            if i > len(A_origin):
                print("i > len(A_origin)")
                repulsion_mask[i][j] = 20

    for helix in [list(helix) for helix in na.helixes]:
        for idx, (i, j) in enumerate(helix):
            target_edge_ratios[i, j] = helix_ratio
            target_edge_ratios[j, i] = helix_ratio

    for knot in [list(na.helixes[idx]) for idx in na.knots]:
        for idx, (i, j) in enumerate(knot):
            dist = int(math.floor(abs((len(knot) - 1) / 2 - idx)))
            # knots_force_ratios[i, j] = math.pow(knots_ratio, dist * 2)
            # knots_force_ratios[j, i] = math.pow(knots_ratio, dist * 2)
            target_edge_ratios[i, j] = math.pow(knots_ratio, dist * 2)
            target_edge_ratios[j, i] = math.pow(knots_ratio, dist * 2)
            print(f"{target_edge_ratios[i, j]=}")

    for iteration in range(iterations):

        displacement *= 0
        centers = find_centers(pos, loops)

        repulsion_force *= 0

        for center_idx, center in enumerate(centers):
            r0 = get_loop_radius(l0, len(loops[center_idx]))
            # print(f"{center_idx} - {r0=} - {center=}")
            for point_idx in loops[center_idx]:
                delta = pos[point_idx] - center
                distance = np.linalg.norm(delta)
                distance = np.where(distance < 0.001, 0.001, distance)

                center_force[point_idx] = (-2) * (1 - r0 / distance) * delta

        # print(f"{center_force=}")

        for i in range(len(A)):

            elastic_force *= 0
            repulsion_force *= 0

            delta = (pos[i] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.001, 0.001, distance)

            for idx1, connected_point_idx in enumerate(A[i]):
                if connected_point_idx >= len(A_origin):
                    continue

                marker_points_ratio = 1.
                if i >= len(A_origin):
                    marker_points_ratio = 50

                elastic_force += (
                        (-2) * (1 - l0 * target_edge_ratios[i][connected_point_idx] / distance[connected_point_idx])
                        * (pos[i] - pos[connected_point_idx]) * marker_points_ratio
                )

            repulsion_force = (repulsion_mask[i] * np.power(distance, -3) * delta).sum(axis=1)

            # print(f"{b*repulsion_force=}")
            displacement[i] = (elastic_inf * elastic_force + repulsion_inf * repulsion_force)

        displacement += center_inf * center_force
        displacement = displacement * time_step

        for idx, d in enumerate(displacement):
            if np.linalg.norm(d) > l0 / 1.3:
                displacement[idx] = d * ((l0 / 2) / np.linalg.norm(d))
                # print(np.linalg.norm(displacement[idx]))

        pos += displacement

    return pos, A


def compute_graph(sequence):
    na = NA(sequence)
    adj = na.get_adjacency()
    adj_list = adjacency_matrix_to_list(adj)
    G = nx.from_dict_of_lists(adj_list)

    pos = nx.spring_layout(G, dim=3, iterations=2500)
    pos = np.array([value for value in pos.values()])

    pos, adj_list = optimize(
        na,
        pos,
        elastic_inf=10,
        repulsion_inf=0.01,
        center_inf=40,
        time_step=0.0001,
        iterations=200,
        knots_ratio=1.3,
        helix_ratio=1.2
    )

    return pos.tolist(), adj_list


def run_script_with_markers(structure_str):
    pos, adj_list = compute_graph(structure_str)
    return json.dumps({'pos': pos, 'adj': adj_list})
