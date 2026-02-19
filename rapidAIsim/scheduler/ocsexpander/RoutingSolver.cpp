#include "RoutingSolver.h"
#include <pybind11/stl.h>
#include <iostream>
//#include <ranges>

void RoutingSolver::generateIntraPodTable() {
    // 生成从nic到leaf的路由和从leaf到nic的路由

    if (is_rail_optimized) {
        // 下面的for循环生成rail-optimized的路由表
        for (int i = 0; i < nic_num; i++) {
            int nic_server_index = i / nic_num_per_server;  // nic所在server编号
            int nic_rail_index = nic_server_index / leaf_up_port_num;  // nic所在rail编号
            int leaf_index = nic_rail_index * nic_num_per_server + i % nic_num_per_server;
            leaf_index += leaf_start_id;
            intraPodUpTable[i].push_back(leaf_index);
            intraPodDownTable[leaf_index][i].push_back(i);
            connection_info_list.push_back({i, leaf_index, 1});
            connection_info_list.push_back({leaf_index, i, 1});
        }
    } else {
        // 下面的for循环生成non-rail-optimized的路由表
        for (int i = 0; i < nic_num; i++) {
            int leaf_nic_num = nic_num / leaf_num;  // 每个leaf上的nic数量
            int leaf_index = i / leaf_nic_num + leaf_start_id;
            intraPodUpTable[i].push_back(leaf_index);
            intraPodDownTable[leaf_index][i].push_back(i);
            connection_info_list.push_back({i, leaf_index, 1});
            connection_info_list.push_back({leaf_index, i, 1});
        }
    }

    // 生成leaf到spine的路由和从spine到leaf的路由
    for (int i = 0; i < leaf_num; i++) {
        int leaf_pod_index = i / (leaf_num / pod_num);  // leaf所在pod编号
        int leaf_index = i + leaf_start_id;  // leaf编号
        for (int j = 0; j < spine_num_per_pod * leaf_spine_link_num; j+=leaf_spine_link_num) {
            int spine_index = j / leaf_spine_link_num + spine_start_id + leaf_pod_index * spine_num_per_pod;  // spine编号
            // std::cout << "leaf_index: " << leaf_index << ", spine_index: " << spine_index << std::endl;
            intraPodUpTable[leaf_index].push_back(spine_index);
            connection_info_list.push_back({leaf_index, spine_index, leaf_spine_link_num});
            connection_info_list.push_back({spine_index, leaf_index, leaf_spine_link_num});
            for (auto &dst : intraPodDownTable[leaf_index]) {  // 查询leaf到nic的路由，添加到spine到nic的下一跳中
                for (int i = 0; i < leaf_spine_link_num; i++) {
                    intraPodDownTable[spine_index + i][dst.first].push_back(leaf_index);
                }
            }
            // for (auto dst : intraPodDownTable[leaf_index] | std::views::keys) {  // 查询leaf到nic的路由，添加到spine到nic的下一跳中
            //     intraPodDownTable[spine_index][dst].push_back(leaf_index);
            // }
        }
    }
}

void RoutingSolver::generateInterPodTable(py::array_t<int> &x_ijkt) {
    // 生成spine间的直连路由
    py::detail::unchecked_mutable_reference<int, 4> x_ijkt_view = x_ijkt.mutable_unchecked<4>();
    for (int i = 0; i < pod_num; i++) {
        for (int j = 0; j < pod_num; j++) {
            if (i == j) {
                continue;
            }
            for (int t = 0; t < spine_num_per_pod; t++) {
                int sum_link_num = 0;
                int src_spine_index = t + spine_start_id + i * spine_num_per_pod;
                int dst_spine_index = t + spine_start_id + j * spine_num_per_pod;
                for (int k = 0; k < spine_up_port_num; k++) {
                    sum_link_num += x_ijkt_view(i, j, k, t);
                }
                if (sum_link_num > 0) {
                    interPodRoutingTable[src_spine_index][j].push_back(dst_spine_index);
                    interPodWeightedDirectRoutingTable[src_spine_index][j].emplace_back(dst_spine_index, sum_link_num);
                    // std::cout << i << "->" << src_spine_index << "->" << dst_spine_index << "->" << j << ": " << sum_link_num << std::endl;
                    connection_info_list.push_back({src_spine_index, dst_spine_index, sum_link_num});
                }
            }
        }
    }
    // 生成spine间的两跳路由
    for (auto &[src_spine_index, src_spine_routing_table] : interPodWeightedDirectRoutingTable) {
        int src_pod = (src_spine_index - spine_start_id) / spine_num_per_pod;
        for (auto &[mid_pod, first_hop_list] : src_spine_routing_table) {
            for (auto &[mid_spine, link_num] : first_hop_list) {
                for (auto &[dst_pod, second_hop_list]: interPodWeightedDirectRoutingTable[mid_spine]) {
                    if (dst_pod == src_pod || dst_pod == mid_pod) {
                        continue;
                    }
                    for (auto &[dst_spine_index, link_num2] : second_hop_list) {
                        float min_link_num = std::min(link_num, link_num2);
                        interPodWeightedTwoHopRoutingTable[src_spine_index][dst_pod].emplace_back(mid_spine, min_link_num);
                    }
                }
            }
        }
    }
    // 把两跳路由表的信息汇总，根据link_num计算weight，分别存储到两个表里面。
    for (auto &[src_spine_index, src_spine_routing_table] : interPodWeightedTwoHopRoutingTable) {
        for (auto &[dst_pod, next_hop_list] : src_spine_routing_table) {
            // 计算weight
            float sum_link_num = 0;
            for (auto &next_hop_direct : interPodWeightedDirectRoutingTable[src_spine_index][dst_pod]) {
                sum_link_num += next_hop_direct.second;
            }
            sum_link_num *= init_alpha;
            for (auto &next_hop_two_hop : next_hop_list) {
                sum_link_num += next_hop_two_hop.second;
            }
            for (auto &it : interPodWeightedDirectRoutingTable[src_spine_index][dst_pod]) {
                it.second = it.second * init_alpha / sum_link_num;
            }
            for (auto &it : next_hop_list) {
                it.second /= sum_link_num;
            }
        }
    }
}

void RoutingSolver::generateRoutingTable(py::array_t<int> &x_ijkt) {
    generateIntraPodTable();
    generateInterPodTable(x_ijkt);
}

RoutingSolver::RoutingSolver(const int pod_num, const int spine_num, const int nic_num, const int server_num_per_pod,
                             const int spine_up_port_num, const int leaf_spine_link_num, const float init_alpha,
                             const bool is_rail_optimized)
        : pod_num(pod_num), spine_num(spine_num), nic_num(nic_num), server_num_per_pod(server_num_per_pod),
          spine_up_port_num(spine_up_port_num), leaf_spine_link_num(leaf_spine_link_num), init_alpha(init_alpha),
          is_rail_optimized(is_rail_optimized) {
    leaf_start_id = nic_num;
    leaf_num = spine_up_port_num * pod_num / leaf_spine_link_num;
    spine_start_id = leaf_start_id + leaf_num;
    leaf_up_port_num = spine_num * leaf_spine_link_num / pod_num;
    spine_num_per_pod = spine_num / pod_num;
    nic_num_per_pod = nic_num / pod_num;
    nic_num_per_server = nic_num_per_pod / server_num_per_pod;
}

PYBIND11_MODULE(routing_solver, m) {
    py::class_<RoutingSolver>(m, "RoutingSolver")
        .def(py::init<int, int, int, int, int, int, float, bool>())
        .def("generate_intra_pod_table", &RoutingSolver::generateIntraPodTable)
        .def("generate_routing_table", &RoutingSolver::generateRoutingTable)
        .def("get_intra_pod_up_table", &RoutingSolver::getIntraPodUpTable)
        .def("get_intra_pod_down_table", &RoutingSolver::getIntraPodDownTable)
        .def("get_inter_pod_routing_table", &RoutingSolver::getInterPodRoutingTable)
        .def("get_inter_pod_weighted_direct_routing_table", &RoutingSolver::getInterPodWeightedDirectRoutingTable)
        .def("get_inter_pod_weighted_twohop_routing_table", &RoutingSolver::getInterPodWeightedTwoHopRoutingTable)
        .def("get_connection_info_list", &RoutingSolver::getConnectionInfoList);
}
