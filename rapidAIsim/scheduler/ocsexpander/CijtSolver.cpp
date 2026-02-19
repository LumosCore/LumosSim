#include <vector>
#include <unordered_map>
#include <chrono>
#include "CijtSolver.h"
#include <ortools/graph/min_cost_flow.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


void CijtSolver::constructNodes() {
    for (int layer_id = 0; layer_id < tree_layer_num; ++layer_id) {
        int start_id = (1 << layer_id) - 1;
        int node_num = 1 << layer_id;
        int ele_num = 1 << (tree_layer_num - layer_id - 1);
        layer_info.push_back({start_id, node_num, ele_num});
    }
    for (int layer_id = tree_layer_num; layer_id < 2 * tree_layer_num; ++layer_id) {
        int start_id = layer_info.back()[0] + layer_info.back()[1];
        int node_num = 1 << (2 * tree_layer_num - layer_id - 1);
        int ele_num = 1 << (layer_id - tree_layer_num);
        layer_info.push_back({start_id, node_num, ele_num});
    }
}

void CijtSolver::setDemand(int curr_it) {
    int global_node_id = 0;
    std::vector<int> line_sum, column_sum(leaf_num, 0);
    line_sum.reserve(leaf_num);

    py::object np_sum = numpy.attr("sum");
    auto c_ij_line_sum = np_sum(c_ij_array, 1).cast<py::array_t<int>>();
    auto c_ij_column_sum = np_sum(c_ij_array, 0).cast<py::array_t<int>>();
    for (int i = 0; i < leaf_num; ++i) {
        line_sum.push_back(c_ij_line_sum.at(i));
        column_sum[i] = c_ij_column_sum.at(i);
    }
    for (int curr_layer_id = 0; curr_layer_id < tree_layer_num; curr_layer_id++) {
        int node_num = layer_info[curr_layer_id][1];
        int ele_num = layer_info[curr_layer_id][2];
        for (int it = 0; it < node_num; ++it) {
            int sum_cij = 0;
            for (int i = it * ele_num; i < (it + 1) * ele_num; ++i) {
                sum_cij += line_sum[i];
            }
            int floor_flow_sum_num = sum_cij / (spine_num_per_pod - curr_it);
            int ceil_flow_sum_num = floor_flow_sum_num + (sum_cij % (spine_num_per_pod - curr_it) > 0 ? 1 : 0);
            floor_flow_sum_num_list[global_node_id] = floor_flow_sum_num;
            ceil_flow_sum_num_list[global_node_id] = ceil_flow_sum_num;
            global_node_id++;
        }
    }
    for (int curr_layer_id = tree_layer_num; curr_layer_id < 2 * tree_layer_num; ++curr_layer_id) {
        int node_num = layer_info[curr_layer_id][1];
        int ele_num = layer_info[curr_layer_id][2];
        for (int it = 0; it < node_num; ++it) {
            int sum_cij = 0;
            for (int i = it * ele_num; i < (it + 1) * ele_num; ++i) {
                sum_cij += column_sum[i];
            }
            int floor_flow_sum_num = sum_cij / (spine_num_per_pod - curr_it);
            int ceil_flow_sum_num = floor_flow_sum_num + (sum_cij % (spine_num_per_pod - curr_it) > 0 ? 1 : 0);
            floor_flow_sum_num_list[global_node_id] = floor_flow_sum_num;
            ceil_flow_sum_num_list[global_node_id] = ceil_flow_sum_num;
            global_node_id++;
        }
    }
}

int CijtSolver::connectGraph(int curr_it, std::vector<int> &supplies, MinCostFlow &min_cost_flow) {
    // connect the graph
    for (int start_id = 0; start_id < total_node_num - 1; ++start_id) {
        if (start_id < layer_info[tree_layer_num - 1][0]) {
            // add the first half of the graph
            // std::cout << "add the first half of graph." << std::endl;
            int end_id = 2 * start_id + 1;
            for (int i = 0; i < 2; i++) {
                end_id += i;
                int capacity = ceil_flow_sum_num_list[end_id] - floor_flow_sum_num_list[end_id];
                min_cost_flow.AddArcWithCapacityAndUnitCost(start_id, end_id, capacity, -1);
                supplies[start_id] -= floor_flow_sum_num_list[end_id];
                supplies[end_id] += floor_flow_sum_num_list[end_id];
            }
        } else if (start_id < layer_info[tree_layer_num][0]) {
            // add the middle part of the graph
            // std::cout << "add the middle part of graph." << std::endl;
            int layer_start_id = layer_info[tree_layer_num - 1][0];
            int end_start_id = layer_info[tree_layer_num][0];
            for (int i = 0; i < leaf_num; i++) {
                int end_id = end_start_id + i;
                int factor = c_ij_array_view(start_id - layer_start_id, i);
                int floor_factor = factor / (spine_num_per_pod - curr_it);
                int ceil_factor = floor_factor + (factor % (spine_num_per_pod - curr_it) > 0 ? 1 : 0);
                int capacity = ceil_factor - floor_factor;
                min_cost_flow.AddArcWithCapacityAndUnitCost(start_id, end_id, capacity, -1);
                supplies[start_id] -= floor_factor;
                supplies[end_id] += floor_factor;
            }
        } else {
            // add the second half of the graph
            // std::cout << "add the second half of graph." << std::endl;
            int end_id = total_node_num - (total_node_num - start_id) / 2;
            if (end_id >= total_node_num) {
                throw std::runtime_error("index out of range.");
            }
            int capacity = ceil_flow_sum_num_list[start_id] - floor_flow_sum_num_list[start_id];
            min_cost_flow.AddArcWithCapacityAndUnitCost(start_id, end_id, capacity, -1);
            supplies[start_id] -= floor_flow_sum_num_list[start_id];
            supplies[end_id] += floor_flow_sum_num_list[start_id];
        }
    }

    // add the final node
    min_cost_flow.AddArcWithCapacityAndUnitCost(total_node_num - 1, 0, 1 << 24, -1);
    for (int i = 0; i < static_cast<int>(supplies.size()); ++i) {
        min_cost_flow.SetNodeSupply(i, supplies[i]);
    }

    int supply_num = 0;
    for (auto &supply: supplies) {
        supply_num += supply;
    }
    if (supply_num != 0) {
        std::cerr << "Supply num is not 0. Construct graph failed." << std::endl;
        return -1;
    }

    return 0;
}

CijtSolver::CijtSolver(const py::array_t<int> &c_ij, int spine_num_per_pod, int leaf_num) :
        spine_num_per_pod(spine_num_per_pod), leaf_num(leaf_num),
        c_ij_array(c_ij) {
    floor_flow_sum_num_list.assign(total_node_num, 0);
    ceil_flow_sum_num_list.assign(total_node_num, 0);
}

int CijtSolver::solveMcfProblem(MinCostFlow &min_cost_flow, py::array_t<int> &c_ijt, int t) {
    auto status = min_cost_flow.Solve();
    if (status != operations_research::SimpleMinCostFlow::OPTIMAL) {
        std::cerr << "Failed to solve the problem. The status is " << status << std::endl;
        return -1;
    }

    // update c_ijt
    auto c_ijt_view = c_ijt.mutable_unchecked<3>();
    for (int i = 0; i < leaf_num; ++i) {
        for (int j = 0; j < leaf_num; ++j) {
            if (i == j) continue;
            int tmp_arc_id = inter_arc_start_id + i * leaf_num + j;
            int new_c_ij_value = static_cast<int>(
                    min_cost_flow.Flow(tmp_arc_id) + c_ij_array_view(i, j) / (spine_num_per_pod - t));
            c_ijt_view(i, j, t) = new_c_ij_value;
            c_ij_array_view(i, j) -= new_c_ij_value;
            if (c_ij_array_view(i, j) < 0) {
                std::cerr << "c_ij[line][column] = " << c_ij_array_view(i, j) << " < 0" << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

py::array_t<int> CijtSolver::solve() {
    auto start = std::chrono::high_resolution_clock::now();
    constructNodes();
    auto stage1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stage1 - start);
    // std::cout << "Construct nodes completed. Time: " << duration1.count() << "ms." << std::endl;

    // prepare the result array
    py::object np_zeros = numpy.attr("zeros");
    auto c_ijt = np_zeros(py::make_tuple(leaf_num, leaf_num, spine_num_per_pod)).cast<py::array_t<int>>();

    // solve the problem for each spine
    for (int t = 0; t < spine_num_per_pod; t++) {

        std::vector<int> supplies(total_node_num, 0);
        operations_research::SimpleMinCostFlow min_cost_flow;

        setDemand(t);
        auto stage2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stage2 - start);
        // std::cout << "Iteration " << t << " set demand completed. Time: "
        //           << duration2.count() << "ms" << std::endl;

        int status = connectGraph(t, supplies, min_cost_flow);
        auto stage3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(stage3 - start);
        if (status == -1)
            throw std::runtime_error("connectGraph exited with code -1.");
        // std::cout << "Iteration " << t << " connect graph completed. Time: "
        //           << duration3.count() << "ms" << std::endl;

        status = solveMcfProblem(min_cost_flow, c_ijt, t);
        auto stage4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(stage4 - start);
        if (status == -1)
            throw std::runtime_error("solveMcfProblem exited with code -1.");
        // std::cout << "Iteration " << t << " solve mcf problem completed. Time:"
        //           << duration4.count() << "ms" << std::endl;
    }
    return c_ijt;
}

PYBIND11_MODULE(cijt_solver, m) {
    py::class_<CijtSolver>(m, "CijtSolver")
            .def(py::init<py::array_t<int>, int, int>())
            .def("solve", &CijtSolver::solve);
}
