import math
import sys
import json

import numpy as np
import networkx as nx
from flask import Response, jsonify

from naskit import *


def find_centers(pos, loops):
    centers = []
    for l in loops:
        coords = np.array([0., 0., 0.])
        for idx in l:
            coords += pos[idx]
        centers.append(coords / len(l))

    ! много написано. Нужные координаты можно взять листом индексов
    ! centers.append(pos[l.nts].mean(0))
    ! а дальше вообще в компрехеншн перевести - return [pos[l.nts].mean(0) for l in loops] - одна строка
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
    return target_len / (2 * math.sin(math.pi / (points_number)))


def get_avg_connection_number(pos, A, loop_points_idxes):
    ! это какое-то большое извращение. Если надо пробежаться по комплиментарным связям используй na.pairs и получишь пары индексов
    ! число ковалентных связей в цепочке всегда одинаковое = N-1, где N - число нуклеотидов. Зачем тут делать каунтеры неясно
    ! эта функция должна принимать na, координаты и содержать два маленьких цикла.
    l0 = 0  # fuck(((((
    connection_number = 0
    not_loop_connection_len = 0
    not_loop_connection_number = 0

    for i in range(len(A)):
        for point_idx in A[i]:
            if i == point_idx:
                continue

            l0 += np.linalg.norm(pos[i] - pos[point_idx])
            if i not in loop_points_idxes or point_idx not in loop_points_idxes:
                not_loop_connection_len += l0
                not_loop_connection_number += 1
            connection_number += 1

    l0 /= (connection_number)
    not_loop_connection_len /= (not_loop_connection_number)

    # print(f"{l0=}")
    # print(f"{not_loop_connection_len=}")

    return l0, not_loop_connection_len


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
    ! дальше нужно работать с матрицами, никаких циклов по листам
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

    ! громоздкая структура
    loop_points_idxes = []
    for l in na.loops:
        loop_points_idxes.extend(l.nts)

    l0, not_loop_connection = get_avg_connection_number(pos, A, loop_points_idxes)

    pos /= l0
    l0 = 1

    x, y, z = zip(*pos)
    ! зачем тут глобалка? она даже больше нигде не используется в файле
    global max_value
    max_value = max(x + y + z, key=abs)

    repulsion_mask = np.ones((len(A), len(A)))
    target_edge_ratios = np.ones((len(A), len(A)))
    displacement = np.zeros((len(A), 3))
    center_force = np.zeros((len(A), 3))
    elastic_force = np.array([0., 0., 0.])
    repulsion_force = np.array([0., 0., 0.])

    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                # print("i == j")
                repulsion_mask[i][j] = 0
                continue
            if j > len(A_origin):
                # print("j > len(A_origin)")
                repulsion_mask[i][j] = 0
                continue
            if i > len(A_origin):
                # print("i > len(A_origin)")
                repulsion_mask[i][j] = 20

    ! про итерирование по хеликсу уже написал
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
            # print(f"{target_edge_ratios[i, j]=}")

    for iteration in range(iterations):

        displacement *= 0
        centers = find_centers(pos, loops)
        l0, not_loop_connection = get_avg_connection_number(pos, A, loop_points_idxes)

        repulsion_force *= 0

        ! этот цикл приемлем, ок
        for center_idx, center in enumerate(centers):
            # r0 = get_loop_radius(not_loop_connection, len(loops[center_idx]))
            r0 = get_loop_radius(l0, len(loops[center_idx]))
            # print(f"{center_idx} - {r0=} - {center=}")

            ! Этого цикла быть не должно
            ! Ты должен взять матрицу координат петли pos[loop.nts] и работать с каждой точкой одновременно
            for point_idx in loops[center_idx]:
                delta = pos[point_idx] - center
                distance = np.linalg.norm(delta)
                ! зачем использовать такую усложненную конструкцию чтобы клипнуть значение?
                ! max(0.001, distance) сделает тоже самое но в 100 раз понятнее
                distance = np.where(distance < 0.001, 0.001, distance)

                ! значения сил потом можно вставить в конкретный участок матрицы, т.к. loop.nts выдает индексы по порядку
                ! center_force[loop.nts[0]:loop.nts[-1]] = посчитанные силы
                ! loop.nts надо сохранить отдельно, т.к. это проперти не кешируется в naskit
                center_force[point_idx] = (-2) * (1 - r0 / distance) * delta

        # print(f"{center_force=}")

        ! это цикл по всем нуклеотидам в цепочке и по их соседям - слишком долго
        ! нужно сделать через матрицы сразу на все точки
        
        for i in range(len(A)):

            elastic_force *= 0
            repulsion_force *= 0

            ! попарные дистанции - np.linalg.norm(pos.reshape(-1, 1, 3) - pos, axis=-1)
            delta = (pos[i] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            ! np.clip(distance, 0.001, np.inf)
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
        # if iteration%25==0:
        #     print(iteration)

    return pos, A


def compute_graph(sequence):
    na = NA(sequence)
    adj = na.get_adjacency()

    adj_list = adjacency_matrix_to_list(adj)
    G = nx.from_dict_of_lists(adj_list)

    knots_list = []
    ! итерироваться можно прямо по хеликсу, не переводя в лист лишний раз
    for h in na.knot_helixes:
        for i, j in h:
            ...
    
    for knot in [list(na.helixes[idx]) for idx in na.knots]:
        for i, j in knot:
            knots_list.append(i)
            knots_list.append(j)

    # print(f"Start spring layout for {sequence}")
    ! число итераций должно подбираться в зависимости от размера структуры или числа связей в ней
    pos = nx.spring_layout(G, dim=3, iterations=5000)
    pos = np.array([value for value in pos.values()])

    # print(f"Start optimize for {sequence}")
    ! число итераций должно подбираться в зависимости от размера структуры или числа связей в ней
    ! 2000 это потолок
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
