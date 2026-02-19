import os
import random
from py2opt.routefinder import RouteFinder
from py2opt.utils import GeographicalPositionTest
import time
import numpy as np


def test_return_to_begin(pod_num, cur_real_link_demand, require_ring_nums):
    res = []
    start_time = time.time()
    # dist_mat = np.zeros((pod_num, pod_num), dtype=int)
    # for i in range(pod_num):
    #     for j in range(pod_num):
    #         dist_mat[i,j] = port_per_port - cur_real_link_demand[i,j]
    dist_mat = cur_real_link_demand
    total_dis = 0
    for ring_id in range(require_ring_nums):
        cities_names = [i for i in range(pod_num)]
        route_finder = RouteFinder(dist_mat, cities_names, iterations=10, return_to_begin=False, verbose=False)
        best_distance_2, _ = route_finder.solve()
        print(best_distance_2)
        print(_)
        res.append(_)
        for tmp_pod_id in range(len(_)):
            start_pod_id = _[tmp_pod_id]
            end_pod_id = _[(1+tmp_pod_id)%pod_num]
            dist_mat[start_pod_id][end_pod_id] -= 1     
        total_dis += best_distance_2
    print("start flow demand time cost",total_dis)
    print(time.time() - start_time)
        
