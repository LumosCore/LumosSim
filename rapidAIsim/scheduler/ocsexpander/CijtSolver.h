#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <deque>
#include <cstring>
#include "ortools/graph/min_cost_flow.h"
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using MinCostFlow = operations_research::SimpleMinCostFlow;

class CijtSolver {
public:
    const int spine_num_per_pod;
    const int leaf_num;
    py::array_t<int> c_ij_array;

    void constructNodes();

    void setDemand(int curr_it);

    int connectGraph(int curr_it, std::vector<int> &supplies, MinCostFlow &min_cost_flow);

    CijtSolver(const py::array_t<int> &c_ij, int spine_num_per_pod, int leaf_num);

    int solveMcfProblem(MinCostFlow &min_cost_flow, py::array_t<int> &c_ijt, int t);

    py::array_t<int> solve();

private:
    int tree_layer_num = (int) log2(leaf_num) + 1;
    int total_node_num = 2 * ((1 << tree_layer_num) - 1);
    int inter_arc_start_id = (1 << tree_layer_num) - 2;
    std::vector<std::deque<std::vector<int>>> layer_list_map;
    std::vector<std::vector<int>> layer_info;
    std::vector<int> floor_flow_sum_num_list;
    std::vector<int> ceil_flow_sum_num_list;
    py::detail::unchecked_mutable_reference<int, 2> c_ij_array_view = c_ij_array.mutable_unchecked<2>();
    py::gil_scoped_acquire acquire;
    py::object numpy = py::module::import("numpy");
};