#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// dst_node_index -> next_hop_list
using RoutingTable = std::unordered_map<int, std::vector<int> >;

// dst_node_index -> (next_hop, weight)_list
using WeightedRoutingTable = std::unordered_map<int, std::vector<std::pair<int, float> > >;

class RoutingSolver {
public:
    RoutingSolver(int pod_num, int spine_num, int nic_num, int server_num_per_pod, int spine_up_port_num,
                  int leaf_spine_link_num, float init_alpha, bool is_rail_optimized);

    void generateRoutingTable(py::array_t<int> &x_ijkt);

    void generateIntraPodTable();

    void generateInterPodTable(py::array_t<int> &x_ijkt);

    std::unordered_map<int, std::vector<int> > getIntraPodUpTable() { return intraPodUpTable; }

    std::unordered_map<int, RoutingTable> getIntraPodDownTable() { return intraPodDownTable; }

    std::unordered_map<int, RoutingTable> getInterPodRoutingTable() { return interPodRoutingTable; }

    std::unordered_map<int, WeightedRoutingTable> getInterPodWeightedDirectRoutingTable() { return interPodWeightedDirectRoutingTable; }

    std::unordered_map<int, WeightedRoutingTable> getInterPodWeightedTwoHopRoutingTable() { return interPodWeightedTwoHopRoutingTable; }

    std::vector<std::vector<int>> getConnectionInfoList() { return connection_info_list; }

private:
    int pod_num;
    int spine_num;
    int nic_num;
    bool is_rail_optimized;

    int server_num_per_pod;
    int nic_num_per_pod;
    int nic_num_per_server;
    int leaf_num;
    int leaf_up_port_num;
    int spine_up_port_num;
    int spine_num_per_pod;
    int leaf_spine_link_num;

    int leaf_start_id;
    int spine_start_id;

    float init_alpha;

    RoutingTable intraPodUpTable;
    std::unordered_map<int, RoutingTable> intraPodDownTable;
    std::unordered_map<int, RoutingTable> interPodRoutingTable;
    std::unordered_map<int, WeightedRoutingTable> interPodWeightedDirectRoutingTable;
    std::unordered_map<int, WeightedRoutingTable> interPodWeightedTwoHopRoutingTable;
    std::vector<std::vector<int>> connection_info_list;
};
