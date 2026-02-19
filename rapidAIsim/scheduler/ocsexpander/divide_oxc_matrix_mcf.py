"""
该模块用于将对称的链路需求矩阵分解为两个互为转置的矩阵之和。
"""
import math
from ortools.linear_solver import pywraplp
import numpy as np
from ortools.graph.python import min_cost_flow
import math


def solve(C_ij, pod_num):
    if not isinstance(C_ij, np.ndarray):
        C_ij = np.array(C_ij)
    arc_capacity_k0_map = {}
    arc_cost_k0_map = {}
    C_id = 0
    for i in range(pod_num):
        for j in range(pod_num):
            if i < j:
                A_id = round(pod_num*(pod_num-1)/2 + C_id)
                A_id_T = round(pod_num*(pod_num-1) + C_id)
                sum_j_A_i = round(pod_num*(pod_num-1)/2*3 + i)
                sum_j_A_j = round(pod_num*(pod_num-1)/2*3 + j)
                arc_capacity_k0_map[f'{C_id}_{A_id}'] = [1]
                arc_cost_k0_map[f'{C_id}_{A_id}'] = [0]
                arc_capacity_k0_map[f'{C_id}_{A_id_T}'] = [1]
                arc_cost_k0_map[f'{C_id}_{A_id_T}'] = [0]
                arc_capacity_k0_map[f'{A_id}_{sum_j_A_i}'] = [math.ceil(C_ij[i,j]/2)]
                arc_cost_k0_map[f'{A_id}_{sum_j_A_i}'] = [0]
                arc_capacity_k0_map[f'{A_id_T}_{sum_j_A_j}'] = [math.ceil(C_ij[i,j]/2)]
                arc_cost_k0_map[f'{A_id_T}_{sum_j_A_j}'] = [0]
                C_id = C_id + 1
    #math.floor(C_ij[i,j]/2)
    start_nodes = []  # 在图中，supply node为0->pod_num*spine_num_per_pod - 1
    end_nodes = []
    capacities = []
    unit_costs = []
    supplies = []
    supplies_map = {}

    global_arc_id = 0
    i_j_t_localarc_2_globalarc_map = {}
    
    for i in range(pod_num):
        sum_j_A_ij = round(pod_num*(pod_num-1)/2*3) + i
        supplies_map[f'{sum_j_A_ij}'] = - round(np.sum(C_ij[i,:]/2))
        # print(i,np.sum(C_ij[i,:]/2))
    
    C_id = 0
    for i in range(pod_num):
        for j in range(pod_num):
            if i < j:
                A_id = round(pod_num*(pod_num-1)/2 + C_id)
                A_id_T = round(pod_num*(pod_num-1) + C_id)
                sum_j_A_i = round(pod_num*(pod_num-1)/2*3 + i)
                sum_j_A_j = round(pod_num*(pod_num-1)/2*3 + j)
                supplies_map[f'{C_id}'] = math.ceil(C_ij[i,j]/2) - math.floor(C_ij[i,j]/2)
                supplies_map[f'{A_id}'] = math.floor(C_ij[i,j]/2)
                supplies_map[f'{A_id_T}'] = math.floor(C_ij[i,j]/2)

                for arc_id in range(len(arc_capacity_k0_map[f'{C_id}_{A_id}'])):
                    if arc_capacity_k0_map[f'{C_id}_{A_id}'][arc_id] > 0:
                        start_nodes.append(C_id)
                        end_nodes.append(A_id)
                        capacities.append(arc_capacity_k0_map[f'{C_id}_{A_id}'][arc_id])
                        unit_costs.append(arc_cost_k0_map[f'{C_id}_{A_id}'][arc_id])
                        i_j_t_localarc_2_globalarc_map[f'{C_id}_{A_id}_{arc_id}_0'] = global_arc_id
                        global_arc_id += 1
                        
                for arc_id in range(len(arc_capacity_k0_map[f'{C_id}_{A_id_T}'])):
                    if arc_capacity_k0_map[f'{C_id}_{A_id_T}'][arc_id] > 0:
                        start_nodes.append(C_id)
                        end_nodes.append(A_id_T)
                        capacities.append(arc_capacity_k0_map[f'{C_id}_{A_id_T}'][arc_id])
                        unit_costs.append(arc_cost_k0_map[f'{C_id}_{A_id_T}'][arc_id])
                        i_j_t_localarc_2_globalarc_map[f'{C_id}_{A_id_T}_{arc_id}_0'] = global_arc_id
                        global_arc_id += 1
                        
                for arc_id in range(len(arc_capacity_k0_map[f'{A_id}_{sum_j_A_i}'])):
                    if arc_capacity_k0_map[f'{A_id}_{sum_j_A_i}'][arc_id] > 0:
                        start_nodes.append(A_id)
                        end_nodes.append(sum_j_A_i)
                        capacities.append(arc_capacity_k0_map[f'{A_id}_{sum_j_A_i}'][arc_id])
                        unit_costs.append(arc_cost_k0_map[f'{A_id}_{sum_j_A_i}'][arc_id])
                        i_j_t_localarc_2_globalarc_map[f'{A_id}_{sum_j_A_i}_{arc_id}_0'] = global_arc_id
                        global_arc_id += 1
                        
                for arc_id in range(len(arc_capacity_k0_map[f'{A_id_T}_{sum_j_A_j}'])):
                    if arc_capacity_k0_map[f'{A_id_T}_{sum_j_A_j}'][arc_id] > 0:
                        start_nodes.append(A_id_T)
                        end_nodes.append(sum_j_A_j)
                        capacities.append(arc_capacity_k0_map[f'{A_id_T}_{sum_j_A_j}'][arc_id])
                        unit_costs.append(arc_cost_k0_map[f'{A_id_T}_{sum_j_A_j}'][arc_id])
                        i_j_t_localarc_2_globalarc_map[f'{A_id_T}_{sum_j_A_j}_{arc_id}_0'] = global_arc_id
                        global_arc_id += 1
                    
                C_id = C_id + 1

    for tmp_id in range(round(pod_num*(pod_num-1)/2*3+pod_num)):
        supplies.append(supplies_map[f'{tmp_id}'])
        
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow_ = min_cost_flow.SimpleMinCostFlow()
    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow_.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i],
                                                            capacities[i], unit_costs[i])

    # print("supply:")
    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow_.set_node_supply(i, supplies[i])

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow_.solve() == min_cost_flow_.OPTIMAL:
        A_i_j_solution = np.zeros((pod_num, pod_num), dtype=int)
        A_T_ij_solution = np.zeros((pod_num, pod_num), dtype=int)
        C_id = 0
        for i in range(pod_num):
            for j in range(pod_num):
                if i < j:
                    A_id = round(pod_num*(pod_num-1)/2 + C_id)
                    A_id_T = round(pod_num*(pod_num-1) + C_id)
                    sum_j_A_i = round(pod_num*(pod_num-1)/2*3 + i)
                    sum_j_A_j = round(pod_num*(pod_num-1)/2*3 + j)
                    
                    for arc_id in range(len(arc_capacity_k0_map[f'{C_id}_{A_id}'])):
                        if arc_capacity_k0_map[f'{C_id}_{A_id}'][arc_id] > 0:
                            global_arc_id = i_j_t_localarc_2_globalarc_map[f'{C_id}_{A_id}_{arc_id}_0']
                            A_i_j_solution[i, j] += min_cost_flow_.flow(global_arc_id) + supplies_map[f'{A_id}']
                            A_T_ij_solution[j, i] += min_cost_flow_.flow(global_arc_id) + supplies_map[f'{A_id}']
                            
                    for arc_id in range(len(arc_capacity_k0_map[f'{C_id}_{A_id_T}'])):
                        if arc_capacity_k0_map[f'{C_id}_{A_id_T}'][arc_id] > 0:
                            global_arc_id = i_j_t_localarc_2_globalarc_map[f'{C_id}_{A_id_T}_{arc_id}_0']
                            A_i_j_solution[j,i] += min_cost_flow_.flow(global_arc_id) + supplies_map[f'{A_id_T}']
                            A_T_ij_solution[i,j] += min_cost_flow_.flow(global_arc_id) + supplies_map[f'{A_id_T}']
                    C_id += 1
                
    else:
        success_MCF = False
        assert success_MCF
        print('There was an issue with the min cost flow input.')

    return A_i_j_solution, A_T_ij_solution


if __name__ == '__main__':
    C_ij = [[0,4,7,3],
            [4,0,2,2],
            [7,2,0,3],
            [3,2,3,0],
            ]

    A_i_j_solution, A_T_ij_solution = solve(C_ij,4)
    print(np.array(C_ij))
    print(A_i_j_solution)
    print(A_T_ij_solution)
