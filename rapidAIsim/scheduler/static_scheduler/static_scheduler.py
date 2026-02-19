from math import ceil
import queue
import math
import rapidAIsim.scheduler.static_scheduler.job as job
import rapidAIsim.scheduler.static_scheduler.switch_resource as switch
import time
from rapidAIsim.scheduler.static_scheduler.utils import *
import logging
from rapidAIsim.scheduler.static_scheduler.job_request import JobRequest
import rapidAIsim.scheduler.static_scheduler.gurobi_solver as gurobi_solver
import rapidAIsim.communication_strategy.detect_butterfly2_conflict as detect_butterfly2_conflict
import gurobipy

logging.basicConfig(level=logging.ERROR)


class StaticScheduler:
    def __init__(self, clos_n, clos_m, opt_segmentation=False,
                 server_num=4):  # spine_switch_num, leaf_switch_num, spine_switch_port_num, leaf_switch_port_num):
        spine_switch_num = clos_n
        leaf_switch_num = clos_m
        spine_switch_port_num = clos_m
        leaf_switch_port_num = clos_n * 2

        gurobi_solver.config_cluster_size(clos_n, clos_m)

        self.spine_num = spine_switch_num
        self.leaf_num = leaf_switch_num
        self.gpu_per_switch = int(leaf_switch_port_num / 2)
        self.gpu_num = self.gpu_per_switch * leaf_switch_num
        self.spine_switch_port_num = spine_switch_port_num
        self.leaf_switch_port_num = leaf_switch_port_num
        self.gpu_per_server = int(self.gpu_num / server_num)
        self.gpu_per_leaf = int(self.gpu_num / self.leaf_num)
        self.server_per_leaf = int(self.gpu_per_leaf / self.gpu_per_server)
        # self.gpu_per_server = 8
        # self.allocated_gpu_num = 0
        self.gpus = []
        for i in range(self.leaf_num):
            self.gpus.append(np.zeros(int(leaf_switch_port_num / 2)))

        logging.debug("cluster size: %d gpus" % self.gpu_num)
        assert spine_switch_port_num == self.leaf_num and int(
            leaf_switch_port_num / 2) == self.spine_num and self.gpu_num == int(self.spine_num * spine_switch_port_num)

        # switches
        self.spine_switches = []
        self.leaf_switches = []
        for i in range(self.spine_num):
            s = switch.switch(spine_switch_port_num)
            # s.take_up_port_by_num(switch_port_num)
            self.spine_switches.append(s)
        for i in range(self.leaf_num):
            s = switch.switch(leaf_switch_port_num)
            # print(int(switch_port_num/2))
            # ports taken by downlinks
            s.take_up_port_by_num(int(leaf_switch_port_num / 2))  # half of the links are downlinks taken by gpu
            self.leaf_switches.append(s)

        # job queue
        self.reqQueue = queue.Queue()
        self.next_req = None

        # allocation method
        self.allocate_methods = ["static_based"]  # 初步看来有static_based和naive两种
        self.allocate_method = self.allocate_methods[0]
        self.allocated_jobs = []

        # statistics
        self.utilization_rate = []
        self.utilization_time = []
        self.ideal_utilization = []

        self.utilization_rate_aggregated = []
        self.served_job_count = 0
        self.allocation_failure_job = []
        self.allocation_failure_job_details = []
        self.failure_logged = False
        self.job_info = {}
        self.free_gpu_before_allocation = 0
        self.l_error = 0
        self.l_retry = 0

        self.allocation_chain = {}
        self.chain_index = 0
        self.larger_allocation = []

        self.time_slot = 0
        self.come_in_time = 0

        self.state_changed = True

        self.allocation_times = 0
        self.total_allocation_sec = 0

        # debug
        self.last_update_time = 0
        self.wasted_time = 0
        self.should_schedule = False

        # opt segmentation
        self.opt_segmentation = opt_segmentation

        # thresh-based policy adjustment
        self.thresh_ratio = 1.0
        self.thresh_line = int(self.thresh_ratio * spine_switch_port_num)
        self.even_allocation_num = 0  # the number of even allocation to all spine servers
        self.to_decrease_thresh = False

        # optimization time limit
        self.time_limit = -1  # reserved

        # conflict detection
        self.conflict_detector = detect_butterfly2_conflict.DetectButterfly2Conflict()
        self.failed_by_conflict = False
        self.conflict_time_point = -1
        self.current_running_task = 0
        self.current_gpu_num = 0
        self.gpu_class_num_map = {0 for i in range(int(math.log2(spine_switch_num * spine_switch_port_num)))}

    def snap_shot(self):
        free_gpus = -np.sum(self.gpus, axis=1) + self.gpu_per_switch
        spine_remaining_ports = [s.free_port_num for s in self.spine_switches]
        leaf_remaining_ports = [l.free_port_num for l in self.leaf_switches]
        return free_gpus, spine_remaining_ports, leaf_remaining_ports

    def set_allocate_method(self, method):
        logging.debug("Setting algorithm:", method)
        assert method in self.allocate_methods
        self.allocate_method = method

    def _translate_gpu_index(self, algo_gpu_allocation):
        output_gpu_indexes = []
        for gpu_entry in algo_gpu_allocation:
            # gpu results
            gpu_index = gpu_entry[0] * self.gpu_per_switch + gpu_entry[1]  # translate to output index
            output_gpu_indexes.append(gpu_index)
        return output_gpu_indexes

    def _translate_allocation(self, algo_gpu_allocation, algo_spine_allocation):
        # output
        allocation_link_mapping = []

        # gpu - leaf links
        for gpu_entry in algo_gpu_allocation:
            # gpu results
            output_gpu_index = gpu_entry[0] * self.gpu_per_switch + gpu_entry[1]  # translate to output index
            leaf_index = gpu_entry[0]  # the leaf switch the gpu locates on
            output_leaf_index = leaf_index + self.gpu_num
            allocation_link_mapping.append((output_gpu_index, output_leaf_index, 1))
            allocation_link_mapping.append((output_leaf_index, output_gpu_index, 1))

        # leaf - spine links
        leaf_to_spine_link_num = {}
        for spine_port_entry in algo_spine_allocation:
            leaf_index = spine_port_entry[1]
            spine_index = spine_port_entry[0]
            if leaf_index not in leaf_to_spine_link_num.keys(): leaf_to_spine_link_num[leaf_index] = {}
            if spine_index not in leaf_to_spine_link_num[leaf_index].keys(): leaf_to_spine_link_num[leaf_index][
                spine_index] = 0
            leaf_to_spine_link_num[leaf_index][spine_index] += 1
        for leaf_index in leaf_to_spine_link_num.keys():
            output_leaf_index = leaf_index + self.gpu_num
            for spine_index in leaf_to_spine_link_num[leaf_index].keys():
                output_spine_index = self.leaf_num + spine_index + self.gpu_num
                allocation_link_mapping.append(
                    (output_leaf_index, output_spine_index, leaf_to_spine_link_num[leaf_index][spine_index]))
                allocation_link_mapping.append(
                    (output_spine_index, output_leaf_index, leaf_to_spine_link_num[leaf_index][spine_index]))

        return allocation_link_mapping

    def print_schedule_statistics(self):
        weights = np.sum(self.utilization_time)
        util_sum = 0
        ideal_util_sum = 0
        for i, t in enumerate(self.utilization_time):
            util_sum += t * self.utilization_rate[i]
            ideal_util_sum += t * self.ideal_utilization[i]

        if weights == 0:
            util_rate = 0
            ideal_util_rate = 1
        else:
            util_rate = util_sum / weights
            ideal_util_rate = ideal_util_sum / weights
        print("--------------schedule statistics------------------")
        print("GPU use rate:", util_rate)
        print("Network segmentation-reduced util rate:", util_rate / ideal_util_rate)
        print("Error: schedule delayed time:", self.wasted_time)
        print("---------------------------------------------------")

        network_fragmentation = []
        for i, t in enumerate(self.utilization_time):
            if self.ideal_utilization[i] == 0:
                network_fragmentation.append(0)
            else:
                network_fragmentation.append(self.utilization_rate[i] / self.ideal_utilization[i])
        return network_fragmentation, self.utilization_time

    def _find_max_possible_allocation(self, allocated_num, queued_jobs):
        max_allocation_num = allocated_num
        for job in queued_jobs:
            req_gpu_num = job.get_task_info()[2]
            if (max_allocation_num + req_gpu_num) <= self.gpu_num:
                max_allocation_num += req_gpu_num
            else:
                break
        return max_allocation_num

    def schedule(self, gpu_num, job_id, sim_time, queued_jobs, strict_clos, check_conflict):
        # temp_z = pow(2, int(math.log2(int(gpu_num))))
        print(f"job {job_id} arrive, required gpu_num is {gpu_num}")
        from rapidAIsim.core.simulator import Simulator
        time_start = time.perf_counter()
        next_req = JobRequest(gpu_num, job_id)

        # debug
        self.should_schedule = False
        self.wasted_time += (sim_time - self.last_update_time)

        # current gpu use rate
        used_gpu_num = np.sum(self.gpus)

        if False and 'best' in Simulator.CONF_DICT:
            new_job = job.job(next_req.request_id)
            self_defined_gpu_list = [[(0, 0), (2, 3)], [(0, 1)], [(0, 2), (0, 3), (1, 0), (1, 1)],
                                     [(1, 2), (1, 3), (2, 0), (2, 1)], [(2, 2)], [(3, 0), (3, 1), (3, 2), (3, 3)]]
            for pair in self_defined_gpu_list[job_id]:
                new_job.add_gpu(pair)
                self.gpus[pair[0]][pair[1]] = 1
            self.allocated_jobs.append(new_job)
            allocate_success, cause_of_failure, algo_gpu_allocation, algo_spine_allocation = (
                True, "", self_defined_gpu_list[job_id], None)
        else:
            allocate_success, cause_of_failure, algo_gpu_allocation, algo_spine_allocation = self.allocate_GPU(
                next_req, strict_clos, check_conflict, sim_time, [])
        if allocate_success:
            all_gpu_index, link_mapping = None, None  # self._translate_cluster_status()#self._translate_allocation_results(algo_gpu_allocation, algo_spine_allocation)
            gpu_indexes = self._translate_gpu_index(algo_gpu_allocation)

            if strict_clos:
                assert algo_spine_allocation is not None
                allocated_link_mapping = self._translate_allocation(algo_gpu_allocation, algo_spine_allocation)
            else:
                if algo_spine_allocation is None:
                    allocated_link_mapping = None
                else:
                    allocated_link_mapping = self._translate_allocation(algo_gpu_allocation, algo_spine_allocation)

            self.utilization_rate.append(used_gpu_num / float(self.gpu_num))
            self.utilization_time.append(sim_time - self.time_slot)
            self.ideal_utilization.append(
                self._find_max_possible_allocation(used_gpu_num, queued_jobs) / float(self.gpu_num))
            self.time_slot = sim_time
            if gpu_num >= 1:
                self.current_running_task += 1
            self.current_gpu_num += gpu_num
        else:
            gpu_indexes = None
            link_mapping = None
            all_gpu_index = None
            allocated_link_mapping = None
        time_end = time.perf_counter()
        time_sum = time_end - time_start
        Simulator.SCHEDULER_TIME_COST[job_id] = 0
        f3 = open('schedule_time_cost.txt', 'a')
        f3.write(str(job_id))
        f3.write(",")
        f3.write(str(time_sum))
        f3.write("\n")
        f3.close()
        comm_pair = None
        if gpu_indexes is not None:
            pow_2_gpu_list = gpu_indexes
            remain_gpu_list = []
            if not power_of_2(len(gpu_indexes)):
                comm_pair = self.fusion_gpu_list(pow_2_gpu_list, remain_gpu_list)
            assert len(gpu_indexes) == gpu_num
        f2 = open('queue_length.txt', 'a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time))
        f2.write("\n")
        f2.close()
        return allocate_success, gpu_indexes, allocated_link_mapping, comm_pair, link_mapping  # allocate_success, gpu_indexes, link_mapping

    def update_finished_job(self, job_id, sim_time, queued_jobs):
        move_flag = False

        if not self.should_schedule:
            self.last_update_time = sim_time
            self.should_schedule = True

        # current gpu use rate
        used_gpu_num = np.sum(self.gpus)
        job_gpu_num = 0
        for i, ongoing_job in enumerate(self.allocated_jobs):
            if ongoing_job.id == job_id:
                # release resources
                # print(ongoing_job.allocated_gpus)
                leaf_switch_indexs = set()
                for gpu_index in ongoing_job.allocated_gpus:
                    # self.allocated_gpu_num -= 1
                    # release gpu
                    assert self.gpus[gpu_index[0]][gpu_index[1]] == 1
                    self.gpus[gpu_index[0]][gpu_index[1]] = 0
                    leaf_switch_indexs.add(gpu_index[0])
                job_gpu_num = len(ongoing_job.allocated_gpus)

                # release leaf switches
                for li in leaf_switch_indexs:
                    if ongoing_job.mini_clos_m > 1:
                        self.leaf_switches[li].free_port_by_num(ongoing_job.mini_clos_n)
                # release spine switches
                for si in ongoing_job.allocated_spine_switches.keys():
                    for li in leaf_switch_indexs:
                        self.spine_switches[si].free_port_index(li)

                self.allocated_jobs.pop(i)
                move_flag = True

                self.utilization_rate.append(used_gpu_num / float(self.gpu_num))
                self.utilization_time.append(sim_time - self.time_slot)
                self.ideal_utilization.append(
                    self._find_max_possible_allocation(used_gpu_num, queued_jobs) / float(self.gpu_num))
                self.time_slot = sim_time
                break
        if job_gpu_num >= 1:
            self.current_running_task -= 1
        self.current_gpu_num -= job_gpu_num
        if not move_flag:
            logging.error("Update status failed! : invalid job_id")
            exit()
        f2 = open('queue_length.txt', 'a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time))
        f2.write("\n")
        f2.close()

    # if spine switches can provide required links
    def check_spine_allocation(self, leaf_switch_set, clos_n):
        if len(leaf_switch_set) < 2: return True
        # find clos_n spines switches to connect the set
        count = 0
        # spine_check = False
        for ss in self.spine_switches:
            if ss.ports_free(leaf_switch_set):
                count += 1
        if count >= clos_n:
            return True
        else:
            print("spine check failed!")
            # time.sleep(1)
            # assert False
            return False

        # count = 0
        # for ls in self.leaf_switches:
        #    if ls.free_port_num > clos_n:
        #        count += 1

    # if leaf switches have enough uplinks
    def check_leaf_allocation(self, leaf_switch_set, clos_n):
        link_shortage = []
        if len(leaf_switch_set) < 2: return link_shortage
        for li in leaf_switch_set:
            if self.leaf_switches[li].free_port_num < clos_n:
                link_shortage.append(li)
        # assert  len(link_shortage) == 0
        return link_shortage

    # check if allocation leads to link contention
    def check_conflict(self, gpu_indexes):
        global_gpu_indexes = self._translate_gpu_index(gpu_indexes)
        return self.conflict_detector.whether_conflict(100, len(global_gpu_indexes), global_gpu_indexes)

    def chosen_gpu(self, record_server_remain_gpu_map, temp_z, remain_chosen_gpu_num):
        model = gurobipy.Model("SpineStrategy solution")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 300)
        server_chosen_num_map = {}
        x_i = {}
        x_r_i = {}
        temp_x_r_i = {}
        leaf_used_emp_server = {}
        leaf_used_remain_gpu = {}
        leaf_used_map = {}
        for server_id in record_server_remain_gpu_map:
            x_i[server_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1, name='x_i')
            x_r_i[server_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0,
                                            ub=record_server_remain_gpu_map[server_id], name='s_i')
            temp_x_r_i[server_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=4, name='temp_x_r_i')
        for leaf_id in range(self.leaf_num):
            leaf_used_emp_server[leaf_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.server_per_leaf,
                                                         name='l_x_i')
            leaf_used_remain_gpu[leaf_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.gpu_per_leaf,
                                                         name='l_x_r_i')
            leaf_used_map[leaf_id] = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1, name='l_u_i')
        obj_val = model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=40960, name='obj')
        model.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        model.update()

        model.addConstr(gurobipy.quicksum(x_i[i] * self.gpu_per_server for i in record_server_remain_gpu_map) == temp_z)
        model.addConstr(gurobipy.quicksum(x_r_i[i] for i in record_server_remain_gpu_map) == remain_chosen_gpu_num)
        for i in record_server_remain_gpu_map:
            model.addConstr(x_r_i[i] + x_i[i] * self.gpu_per_server <= record_server_remain_gpu_map[i])
            model.addConstr(x_r_i[i] == 2 * temp_x_r_i[i])
            leaf_id = int(i / self.server_per_leaf)
            model.addConstr(leaf_used_map[leaf_id] >= x_i[i])
            model.addConstr(leaf_used_map[leaf_id] * record_server_remain_gpu_map[server_id] >= x_r_i[i])
        for leaf_id in range(self.leaf_num):
            model.addConstr(gurobipy.quicksum(x_i[server_id] for server_id in record_server_remain_gpu_map if
                                              int(server_id / self.server_per_leaf) == leaf_id) == leaf_used_emp_server[
                                leaf_id])
            model.addConstr(gurobipy.quicksum(x_r_i[server_id] for server_id in record_server_remain_gpu_map if
                                              int(server_id / self.server_per_leaf) == leaf_id) == leaf_used_remain_gpu[
                                leaf_id])
            model.addConstr(leaf_used_emp_server[leaf_id] * self.gpu_per_server >= leaf_used_remain_gpu[leaf_id])

        model.addConstr(obj_val >= gurobipy.quicksum(leaf_used_map[leaf_id] for leaf_id in leaf_used_emp_server))
        model.update()
        model.optimize()
        print("finish running")
        # 记录运行结果
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            x_i_sol = model.getAttr('X', x_i)
            x_r_i_sol = model.getAttr('X', x_r_i)
            server_chosen_num_map = {}
            for server_id in x_i_sol:
                if int(x_i_sol[server_id] * self.gpu_per_server) > 0:
                    server_chosen_num_map[server_id] = int(x_i_sol[server_id] * self.gpu_per_server)
            for server_id in x_r_i_sol:
                if int(x_r_i_sol[server_id]) > 0:
                    server_chosen_num_map[server_id] = int(x_r_i_sol[server_id])
            # print("debug x_i_sol", x_i_sol, len(x_i_sol))
            # print("debug x_r_i_sol", x_r_i_sol, len(x_r_i_sol))
            return True, server_chosen_num_map
        else:
            return False, None

    def fusion_gpu_list(self, pow_2_gpu_list, remain_gpu_list):
        communication_pair_list = []
        fus_gpu_list = []
        fus_gpu_list.extend(pow_2_gpu_list)
        fus_gpu_list.extend(remain_gpu_list)
        gpu_global_local_index_map = {}
        for i in range(len(fus_gpu_list)):
            gpu_global_local_index_map[fus_gpu_list[i]] = i
        # print(gpu_global_local_index_map)
        # print(pow_2_gpu_list, len(pow_2_gpu_list))
        # print(remain_gpu_list, len(remain_gpu_list))
        leaf_pow_2_gpu_list_map = {}
        for pow_2_gpu in pow_2_gpu_list:
            leaf_id = int(pow_2_gpu / self.gpu_per_leaf)
            if leaf_id not in leaf_pow_2_gpu_list_map:
                leaf_pow_2_gpu_list_map[leaf_id] = []
            leaf_pow_2_gpu_list_map[leaf_id].append(pow_2_gpu)
        leaf_remain_gpu_list_map = {}
        temp_index = 0
        for remain_gpu in remain_gpu_list:
            leaf_id = int(remain_gpu / self.gpu_per_leaf)
            if leaf_id not in leaf_remain_gpu_list_map:
                leaf_remain_gpu_list_map[leaf_id] = []
            to_comm_gpu = pow_2_gpu_list[temp_index]
            leaf_remain_gpu_list_map[leaf_id].append(remain_gpu)
            communication_pair_list.append(
                (gpu_global_local_index_map[remain_gpu], gpu_global_local_index_map[to_comm_gpu]))
            temp_index += 1
        # for leaf_id in leaf_remain_gpu_list_map:
        #     for remain_gpu in leaf_remain_gpu_list_map[leaf_id]:
        #         to_comm_gpu = leaf_pow_2_gpu_list_map[leaf_id][0]
        #         communication_pair_list.append((gpu_global_local_index_map[remain_gpu], gpu_global_local_index_map[to_comm_gpu]))
        #         del(leaf_pow_2_gpu_list_map[leaf_id][0])
        return communication_pair_list

    def allocate_GPU(self, next_req: JobRequest, strict_clos, check_conflict, sim_time,
                     banned_server_list=None):
        # print("start allo")
        #### cause of allocation failure
        if banned_server_list is None:
            banned_server_list = [2, 4]
        failure_cause = ""
        self.failed_by_conflict = False

        # self.allocate_method = "arbitrary"

        if self.allocate_method == "static_based":
            # print("Time:", self.time_slot)  # next_req.exec_time)
            # a naive way for allocating GPUs as split clos
            if int((self.gpu_num - np.sum(self.gpus))) < next_req.gpu_num:
                print("no resource0", self.gpu_num - np.sum(self.gpus), next_req.gpu_num)
                return False, failure_cause, None, None

            # decide gpu allocation
            free_gpus = -np.sum(self.gpus, axis=1) + self.gpu_per_switch
            remain_ports = [(self.gpu_per_switch - l.free_port_num) for l in self.leaf_switches]
            request_num = next_req.gpu_num
            print("debug free_gpus", self.gpu_num - np.sum(self.gpus), next_req.gpu_num)
            allocation_success = False
            gpu_indexes = []
            clos_n = -1
            clos_m = -1
            ss_indexes = None
            if not strict_clos:
                gpu_indexes = []
                # simply see if there are enough remaining GPU
                if (self.gpu_num - np.sum(self.gpus)) < next_req.gpu_num:
                    print("no resource0")
                    return False, failure_cause, None, None

                # There are enough GPUs
                distrib_count = 0
                new_job = job.job(next_req.request_id)  # next_req.exec_time)

                # temp_z = pow(2, int(math.log2(int(request_num))))
                # if temp_z == next_req.gpu_num or next_req.gpu_num<self.gpu_per_leaf:

                require_server_num = ceil(request_num / self.gpu_per_server)
                temp_server_remain_gpu_map = {}
                for li in range(self.leaf_num):
                    for j in range(self.gpu_per_switch):  # we do not care about the gpu
                        temp_server_id = int(j / self.gpu_per_server) + self.server_per_leaf * li
                        temp_gpu_id = j + self.gpu_per_switch * li
                        if temp_server_id not in banned_server_list:
                            temp_gpu_id = j + self.gpu_per_switch * li
                            if self.gpus[li][j] == 0:
                                if temp_server_id not in temp_server_remain_gpu_map:
                                    temp_server_remain_gpu_map[temp_server_id] = []
                                temp_server_remain_gpu_map[temp_server_id].append(temp_gpu_id)
                temp_server_remain_gpu_map = dict(
                    sorted(temp_server_remain_gpu_map.items(), key=lambda x: len(x[1]), reverse=False))
                temp_server_remain_gpu_list = []
                for server_id in temp_server_remain_gpu_map:
                    temp_remain_gpu_list = temp_server_remain_gpu_map[server_id]
                    if len(temp_remain_gpu_list) == self.gpu_per_server or len(
                            temp_remain_gpu_list) >= next_req.gpu_num:
                        temp_server_remain_gpu_list.append([server_id, temp_remain_gpu_list])
                if len(temp_server_remain_gpu_list) < require_server_num:
                    print("no resource1", (self.gpu_num - np.sum(self.gpus)), next_req.gpu_num,
                          len(temp_server_remain_gpu_list), require_server_num)
                    return False, failure_cause, None, None
                else:
                    for i in range(require_server_num):
                        for gpu_id in temp_server_remain_gpu_list[i][1]:
                            if distrib_count < next_req.gpu_num:
                                distrib_count += 1
                                gpu_indexes.append(
                                    (int(gpu_id / self.gpu_per_switch), gpu_id % self.gpu_per_switch))
                                new_job.add_gpu((int(gpu_id / self.gpu_per_switch), gpu_id % self.gpu_per_switch))
                    link_conflicted = False
                    if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)
                    if not link_conflicted:
                        self.allocated_jobs.append(new_job)
                        for pair in gpu_indexes:
                            self.gpus[pair[0]][pair[1]] = 1
                        # print("finish allo", self.gpu_num, self.gpu_num - np.sum(self.gpus), next_req.gpu_num)
                        print("gpu_indexes", gpu_indexes)
                        assert len(gpu_indexes) == request_num
                        return True, failure_cause, gpu_indexes, None
                    else:
                        return False, failure_cause, None, None
                # else:
                #     remain_num = request_num - temp_z
                #     temp_server_remain_gpu_map = {}
                #     for li in range(self.leaf_num):
                #         for j in range(self.gpu_per_switch):  # we do not care about the gpu
                #             temp_server_id = int(j/self.gpu_per_server)+self.server_per_leaf*li
                #             temp_gpu_id = j+self.gpu_per_switch*li
                #             if self.gpus[li][j] == 0:
                #                 if temp_server_id not in temp_server_remain_gpu_map:
                #                     temp_server_remain_gpu_map[temp_server_id] = []
                #                 temp_server_remain_gpu_map[temp_server_id].append(temp_gpu_id)
                #     temp_server_remain_gpu_map = dict( sorted(temp_server_remain_gpu_map.items(),key = lambda x:len(x[1]),reverse = False))
                #     record_server_remain_gpu_map = {}
                #     for server_id in temp_server_remain_gpu_map:
                #         temp_remain_gpu_list_num = len(temp_server_remain_gpu_map[server_id])
                #         record_server_remain_gpu_map[server_id] = temp_remain_gpu_list_num
                #     # 根据server信息和一个leaf多少server选择GPU
                #     print("debug remain num", temp_z,remain_num)
                #     allocate_success, server_chosen_num_map = self.chosen_gpu(record_server_remain_gpu_map,temp_z,remain_num)
                #     if not allocate_success:
                #         print("no resource1")
                #         return False, failure_cause, None, None
                #     else:
                #         require_server_num = 0
                #         has_chosen_gpu = 0
                #         temp_server_remain_gpu_list = []
                #         for server_id in server_chosen_num_map:
                #             temp_remain_gpu_list = temp_server_remain_gpu_map[server_id]
                #             if len(temp_remain_gpu_list)>=server_chosen_num_map[server_id]:
                #                 # print("debug server_chosen_num_map[server_id]",server_chosen_num_map[server_id])
                #                 temp_server_remain_gpu_list.append([server_id, temp_remain_gpu_list[:server_chosen_num_map[server_id]]])
                #                 require_server_num += 1
                #                 has_chosen_gpu += server_chosen_num_map[server_id]
                #         print("allocate_success", temp_server_remain_gpu_list, has_chosen_gpu, request_num)
                #         assert has_chosen_gpu == request_num
                #         for i in range(require_server_num):
                #             for gpu_id in temp_server_remain_gpu_list[i][1]:
                #                 if distrib_count< next_req.gpu_num:
                #                     distrib_count+=1
                #                     gpu_indexes.append((int(gpu_id/self.gpu_per_switch), gpu_id%self.gpu_per_switch))
                #                     new_job.add_gpu((int(gpu_id/self.gpu_per_switch), gpu_id%self.gpu_per_switch))
                #         link_conflicted = False
                #         if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)
                #         if not link_conflicted:
                #             self.allocated_jobs.append(new_job)
                #             for pair in gpu_indexes:
                #                 self.gpus[pair[0]][pair[1]] = 1
                #             print("finish allo", self.gpu_num, self.gpu_num - np.sum(self.gpus), next_req.gpu_num)
                #             assert len(gpu_indexes) == request_num
                #             return True, failure_cause, gpu_indexes, None
                #         else:
                #             return False, failure_cause, None, None

                # for i in range(self.leaf_num):
                #     for j in range(self.gpu_per_switch):
                #         if self.gpus[i][j] == 0:
                #             self.gpus[i][j] = 1
                #             distrib_count += 1
                #             # self.allocated_gpu_num += 1
                #             new_job.add_gpu((i, j))
                #             gpu_indexes.append((i, j))
                #             if distrib_count >= next_req.gpu_num:
                #                 link_conflicted = False
                #                 if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)
                #                 if not link_conflicted:
                #                     self.allocated_jobs.append(new_job)
                #                     print("fuck ", gpu_indexes)
                #                     return True, failure_cause, gpu_indexes, None
                #                 else:
                #                     return False, failure_cause, None, None
            else:
                while True:  # use while loop so we can easily break out

                    # first case
                    # print("Request:", request_num)
                    # print("-------------------------p1")
                    if request_num % self.gpu_per_switch == 0:
                        # try to allocate complete tor
                        # print(request_num)
                        # assert request_num % self.gpu_per_switch == 0
                        tor_num = int(request_num / self.gpu_per_switch)
                        tor_indexes = []
                        count = 0
                        for li in range(self.leaf_num):
                            if np.sum(self.gpus[li]) == 0:
                                count += 1
                                tor_indexes.append(li)
                            if count >= tor_num:
                                allocation_success = True
                                clos_n = self.gpu_per_switch
                                clos_m = tor_num
                                for i in tor_indexes:
                                    for j in range(self.gpu_per_switch):
                                        gpu_indexes.append((i, j))
                                assert len(gpu_indexes) > 0
                                break

                    # check if spine allocation holds
                    if allocation_success:
                        leaf_switch_set = set([i[0] for i in gpu_indexes])
                        # print(leaf_switch_set)
                        link_conflicted = False
                        if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)

                        if link_conflicted:
                            self.failed_by_conflict = True
                            # self.conflict_time_point = sim_time
                        if self.check_spine_allocation(leaf_switch_set, clos_n) and not link_conflicted:
                            break
                        else:  # reset
                            allocation_success = False
                            gpu_indexes = []
                            clos_n = -1
                            clos_m = -1

                    if allocation_success:
                        print("allocation1")
                        break

                    # assert False
                    # print("-------------------------p2-local")
                    # locality
                    if request_num < self.gpu_per_switch:
                        # find the leaf switch with the least remaining gpus [todo: ports]
                        indexes = np.argsort(free_gpus)
                        for li in indexes:
                            if free_gpus[li] >= request_num:
                                distrib_count = 0
                                temp_server_remain_gpu_map = {}
                                for j in range(self.gpu_per_switch):  # we do not care about the gpu
                                    temp_server_id = int(j / self.gpu_per_server)
                                    if self.gpus[li][j] == 0:
                                        if temp_server_id not in temp_server_remain_gpu_map:
                                            temp_server_remain_gpu_map[temp_server_id] = []
                                        temp_server_remain_gpu_map[temp_server_id].append(j)
                                temp_server_remain_gpu_list = []
                                for server_id in temp_server_remain_gpu_map:
                                    temp_remain_gpu_list = temp_server_remain_gpu_map[server_id]
                                    temp_server_remain_gpu_list.append([server_id, temp_remain_gpu_list])
                                temp_server_remain_gpu_list.sort(key=lambda x: (len(x[1])), reverse=True)

                                can_deuce = False
                                need_server_num = 1
                                while need_server_num <= int(self.gpu_per_switch / 4):
                                    potentional_server_groupsize_pair_list = []
                                    need_group_each_server = int(request_num / need_server_num)
                                    for potentional_server_pair in temp_server_remain_gpu_list:
                                        temp_remain_gpu_num = len(potentional_server_pair[1])
                                        if temp_remain_gpu_num >= need_group_each_server:
                                            potentional_server_groupsize_pair_list.append(potentional_server_pair[1])
                                    if len(potentional_server_groupsize_pair_list) < need_server_num:
                                        need_server_num *= 2
                                    else:
                                        temp_server_remain_gpu_list.sort(key=lambda x: (len(x[1])), reverse=True)
                                        for temp_flag in range(need_server_num):
                                            temp_gpu_list = potentional_server_groupsize_pair_list[temp_flag]
                                            chosen_num = 0
                                            while chosen_num < need_group_each_server:
                                                temp_gpu_index = temp_gpu_list[0]
                                                del temp_gpu_list[0]
                                                chosen_num += 1
                                                gpu_indexes.append((li, temp_gpu_index))
                                                # print(li, temp_gpu_index,need_server_num,chosen_num,need_group_each_server)
                                                # exit()
                                        can_deuce = True
                                        break

                                if not can_deuce:
                                    while distrib_count < request_num:
                                        assert temp_server_remain_gpu_list[0][1] != []
                                        temp_gpu_index = temp_server_remain_gpu_list[0][1][0]
                                        del temp_server_remain_gpu_list[0][1][0]
                                        distrib_count += 1
                                        gpu_indexes.append((li, temp_gpu_index))
                                        if temp_server_remain_gpu_list[0][1] == []:
                                            del temp_server_remain_gpu_list[0]
                                # for j in range(self.gpu_per_switch):  # we do not care about the gpu
                                #     if self.gpus[li][j] == 0:
                                #         distrib_count += 1
                                #         gpu_indexes.append((li, j))
                                #         if distrib_count >= request_num:
                                #             break

                                assert len(gpu_indexes) > 0
                                allocation_success = True
                                clos_n = request_num
                                clos_m = 1
                                break

                    # no need to check spine allocation
                    if allocation_success:
                        print("allocation1")
                        break

                    # print("-------------------------p3")
                    # cross tor allocation
                    #### First check for perfect clos
                    start_s = int(self.gpu_per_switch)
                    tmp_free_gpus = list(free_gpus)
                    while start_s >= 1:  # < self.gpu_per_switch:
                        qualified = np.array(tmp_free_gpus) > start_s
                        if np.sum(qualified) * start_s >= request_num:
                            tor_num = int(request_num / start_s)
                            assert tor_num > 1
                            # try solution with gurobi
                            gpu_status = -np.sum(self.gpus, axis=1) + self.gpu_per_switch
                            spine_ports_status = self._copy_spine_port_status()

                            allocation_success, clos_n, clos_m, leaf_indexes_selected, gpu_num_selected, spine_indexes_selected = gurobi_solver.allocate_resources_for_given_mn(
                                gpu_status, spine_ports_status, start_s, tor_num)

                            if allocation_success:
                                for i, li in enumerate(leaf_indexes_selected):
                                    allocated = 0
                                    for j in range(self.gpu_per_switch):
                                        if self.gpus[li][j] == 0:
                                            allocated += 1
                                            gpu_indexes.append((li, j))
                                            if allocated >= gpu_num_selected[i]:
                                                break
                                ss_indexes = spine_indexes_selected

                                link_conflicted = False
                                if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)

                                if link_conflicted:
                                    self.failed_by_conflict = True
                                    # self.conflict_time_point = sim_time

                                if not link_conflicted:
                                    break
                                else:
                                    allocation_success = False
                                    gpu_indexes = []
                                    clos_n = -1
                                    clos_m = -1

                            else:
                                allocation_success = False
                                gpu_indexes = []
                                clos_n = -1
                                clos_m = -1
                        start_s = int(start_s / 2)

                    # break

                    if allocation_success:
                        # print(start_s)
                        # assert False
                        print("allocation2")
                        break

                    # print("-------------------------p4")
                    failure_cause = "p4"
                    #### Then check for imbalanced allocation
                    ## logic 1

                    tmp_free_gpus = list(free_gpus)
                    while self.opt_segmentation and True:
                        indexes = np.argsort(np.array(tmp_free_gpus) * (1))
                        sorted_gpus = np.array(tmp_free_gpus)[indexes]
                        gpu_count = 0
                        ls_count = 0
                        max_n = 0
                        for i, li in enumerate(indexes):
                            if sorted_gpus[i] <= 0:
                                # ls_count += 1
                                continue
                            if gpu_count + sorted_gpus[i] >= request_num:
                                to_allocate = request_num - gpu_count
                            else:
                                to_allocate = sorted_gpus[i]
                            if to_allocate > max_n: max_n = to_allocate

                            distrib_count = 0
                            temp_server_remain_gpu_map = {}
                            for j in range(self.gpu_per_switch):  # we do not care about the gpu
                                temp_server_id = int(j / 4)
                                if self.gpus[li][j] == 0:
                                    if temp_server_id not in temp_server_remain_gpu_map:
                                        temp_server_remain_gpu_map[temp_server_id] = []
                                    temp_server_remain_gpu_map[temp_server_id].append(j)
                            temp_server_remain_gpu_list = []
                            for server_id in temp_server_remain_gpu_map:
                                temp_remain_gpu_list = temp_server_remain_gpu_map[server_id]
                                temp_server_remain_gpu_list.append([server_id, temp_remain_gpu_list])

                            can_deuce = False
                            need_server_num = 1
                            while need_server_num <= int(self.gpu_per_switch / 4):
                                potentional_server_groupsize_pair_list = []
                                need_group_each_server = int(to_allocate / need_server_num)
                                for potentional_server_pair in temp_server_remain_gpu_list:
                                    temp_remain_gpu_num = len(potentional_server_pair[1])
                                    if temp_remain_gpu_num >= need_group_each_server:
                                        potentional_server_groupsize_pair_list.append(potentional_server_pair[1])
                                if len(potentional_server_groupsize_pair_list) < need_server_num:
                                    need_server_num *= 2
                                else:
                                    temp_server_remain_gpu_list.sort(key=lambda x: (len(x[1])), reverse=True)
                                    for temp_flag in range(need_server_num):
                                        temp_gpu_list = potentional_server_groupsize_pair_list[temp_flag]
                                        chosen_num = 0
                                        while chosen_num < need_group_each_server:
                                            temp_gpu_index = temp_gpu_list[0]
                                            del temp_gpu_list[0]
                                            chosen_num += 1
                                            gpu_indexes.append((li, temp_gpu_index))
                                    can_deuce = True
                                    break
                            if not can_deuce:
                                while distrib_count < to_allocate:
                                    assert temp_server_remain_gpu_list[0][1] != []
                                    temp_gpu_index = temp_server_remain_gpu_list[0][1][0]
                                    del temp_server_remain_gpu_list[0][1][0]
                                    distrib_count += 1
                                    gpu_indexes.append((li, temp_gpu_index))
                                    if temp_server_remain_gpu_list[0][1] == []:
                                        del temp_server_remain_gpu_list[0]

                            # for j in range(self.gpu_per_switch):
                            #     if self.gpus[li][j] == 0:
                            #         distrib_count += 1
                            #         gpu_indexes.append((li, j))
                            #         if distrib_count >= to_allocate:
                            #             break
                            gpu_count += to_allocate
                            ls_count += 1
                            if gpu_count >= request_num:
                                failure_cause = "pii"
                                allocation_success = True
                                assert len(gpu_indexes) > 0
                                break

                        if allocation_success:
                            clos_n = max_n  # sorted_gpus[index_count - 1]
                            clos_m = ls_count
                            leaf_switch_set = set([i[0] for i in gpu_indexes])

                            link_shortage = self.check_leaf_allocation(leaf_switch_set, clos_n)
                            if len(link_shortage) > 0:
                                for i in link_shortage:
                                    tmp_free_gpus[i] = 0
                                # print(clos_n)
                                allocation_success = False
                                failure_cause = "link_shortage"
                                gpu_indexes = []
                                clos_n = -1
                                clos_m = -1
                                # print(link_shortage)
                                # print(tmp_free_gpus)
                                continue

                            link_conflicted = False
                            if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)

                            if link_conflicted:
                                self.failed_by_conflict = True
                                # self.conflict_time_point = sim_time

                            if self.check_spine_allocation(leaf_switch_set, clos_n) and not link_conflicted:
                                break
                            else:  # reset
                                allocation_success = False
                                failure_cause = "no_spine_port"
                                gpu_indexes = []
                                clos_n = -1
                                clos_m = -1
                        break

                    # end of a loop
                    break

            gg, ss, ll = self.snap_shot()
            # print("gpu:", gg)
            # print("spine:", list(ss))
            # print("leaf", [int(k) for k in ll])
            if allocation_success:
                # log the waiting time caused by conflict checking
                if self.conflict_time_point != -1:
                    conflict_waiting_time = sim_time - self.conflict_time_point
                self.conflict_time_point = -1

                # execute the allocation
                assert len(gpu_indexes) > 0 and clos_m >= 0 and clos_n >= 0

                # create job
                new_job = job.job(next_req.request_id)
                for i in gpu_indexes:
                    new_job.add_gpu(i)
                    self.gpus[i[0]][i[1]] = 1
                new_job.mini_clos_n = clos_n
                new_job.mini_clos_m = clos_m
                self.allocated_jobs.append(new_job)

                # take up switch resources
                leaf_switch_set = set([i[0] for i in gpu_indexes])
                assert clos_m == len(leaf_switch_set)
                # print("leaf sets:", leaf_switch_set)
                if ss_indexes == None:
                    ss_indexes = self.allocate_spine_switches(leaf_switch_set, clos_n)
                else:
                    assert len(ss_indexes) == clos_n
                    for si in ss_indexes:
                        for li in leaf_switch_set:
                            self.spine_switches[si].take_up_port_by_index(li)

                allocated_leaf_spine_links = []
                for si in ss_indexes:
                    new_job.add_spine_switch(si, len(leaf_switch_set))
                    for li in leaf_switch_set:
                        allocated_leaf_spine_links.append((si, li))

                self.allocate_leaf_switches(leaf_switch_set, clos_n)
                return True, failure_cause, gpu_indexes, allocated_leaf_spine_links
            else:
                # log failed_by_conflict timepoint
                if self.failed_by_conflict:
                    self.conflict_time_point = sim_time

                # print("gpu:", list(gg))
                # print("spine:", self._copy_spine_port_status())
                # exit()
                # check no clos allocation
                if not strict_clos:
                    gpu_indexes = []
                    # simply see if there are enough remaining GPU
                    if (self.gpu_num - np.sum(self.gpus)) < next_req.gpu_num:
                        return False, failure_cause, None, None
                    else:
                        distrib_count = 0
                        new_job = job.job(next_req.request_id)  # next_req.exec_time)
                        for i in range(self.leaf_num):
                            for j in range(self.gpu_per_switch):
                                if self.gpus[i][j] == 0:
                                    self.gpus[i][j] = 1
                                    distrib_count += 1
                                    # self.allocated_gpu_num += 1
                                    new_job.add_gpu((i, j))
                                    gpu_indexes.append((i, j))
                                    if distrib_count >= next_req.gpu_num:
                                        link_conflicted = False
                                        if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)
                                        if not link_conflicted:
                                            self.allocated_jobs.append(new_job)
                                            return True, failure_cause, gpu_indexes, None
                                        else:
                                            return False, failure_cause, None, None

                return False, failure_cause, None, None  # allocated_leaf_spine_links

        elif self.allocate_method == "naive":
            # the same as naive, except that oxc enables taks to use idle spine ports freely
            pass

    def _greedy_search_gpu(self, next_req: JobRequest, failure_cause, check_conflict, banned_server_list):
        request_num = next_req.gpu_num
        gpu_indexes = []
        # simply see if there are enough remaining GPU
        if (self.gpu_num - np.sum(self.gpus)) < next_req.gpu_num:
            print("no resource0")
            return False, failure_cause, None, None

        # There are enough GPUs
        distrib_count = 0
        new_job = job.job(next_req.request_id)  # next_req.exec_time)

        # temp_z = pow(2, int(math.log2(int(request_num))))
        # if temp_z == next_req.gpu_num or next_req.gpu_num<self.gpu_per_leaf:

        require_server_num = ceil(request_num / self.gpu_per_server)
        temp_server_remain_gpu_map = {}
        for li in range(self.leaf_num):
            for j in range(self.gpu_per_switch):  # we do not care about the gpu
                temp_server_id = int(j / self.gpu_per_server) + self.server_per_leaf * li
                temp_gpu_id = j + self.gpu_per_switch * li
                if temp_server_id not in banned_server_list:
                    temp_gpu_id = j + self.gpu_per_switch * li
                    if self.gpus[li][j] == 0:
                        if temp_server_id not in temp_server_remain_gpu_map:
                            temp_server_remain_gpu_map[temp_server_id] = []
                        temp_server_remain_gpu_map[temp_server_id].append(temp_gpu_id)
        temp_server_remain_gpu_map = dict(
            sorted(temp_server_remain_gpu_map.items(), key=lambda x: len(x[1]), reverse=False))
        temp_server_remain_gpu_list = []
        for server_id in temp_server_remain_gpu_map:
            temp_remain_gpu_list = temp_server_remain_gpu_map[server_id]
            if len(temp_remain_gpu_list) == self.gpu_per_server or len(
                    temp_remain_gpu_list) >= next_req.gpu_num:
                temp_server_remain_gpu_list.append([server_id, temp_remain_gpu_list])
        if len(temp_server_remain_gpu_list) < require_server_num:
            print("no resource1", (self.gpu_num - np.sum(self.gpus)), next_req.gpu_num,
                  len(temp_server_remain_gpu_list), require_server_num)
            return False, failure_cause, None, None
        else:
            for i in range(require_server_num):
                for gpu_id in temp_server_remain_gpu_list[i][1]:
                    if distrib_count < next_req.gpu_num:
                        distrib_count += 1
                        gpu_indexes.append(
                            (int(gpu_id / self.gpu_per_switch), gpu_id % self.gpu_per_switch))
                        new_job.add_gpu((int(gpu_id / self.gpu_per_switch), gpu_id % self.gpu_per_switch))
            link_conflicted = False
            if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)
            if not link_conflicted:
                self.allocated_jobs.append(new_job)
                for pair in gpu_indexes:
                    self.gpus[pair[0]][pair[1]] = 1
                # print("finish allo", self.gpu_num, self.gpu_num - np.sum(self.gpus), next_req.gpu_num)
                print("gpu_indexes", gpu_indexes)
                assert len(gpu_indexes) == request_num
                return True, failure_cause, gpu_indexes, None
            else:
                return False, failure_cause, None, None

    def _copy_spine_port_status(self):
        spine_port_status = []
        for ss in self.spine_switches:
            spine_port_status.append(list(ss.port_status))
        return spine_port_status

    def allocate_spine_switches(self, leaf_switch_set, clos_n):
        spine_switch_indexes = []
        if len(leaf_switch_set) < 2: return spine_switch_indexes
        for i, ss in enumerate(self.spine_switches):
            if ss.ports_free(leaf_switch_set):
                spine_switch_indexes.append(i)
                for li in leaf_switch_set:
                    self.spine_switches[i].take_up_port_by_index(li)
                if len(spine_switch_indexes) >= clos_n: break
        return spine_switch_indexes

    def allocate_leaf_switches(self, leaf_switch_set, clos_n):
        if len(leaf_switch_set) < 2: return
        for li in leaf_switch_set:
            self.leaf_switches[li].take_up_port_by_num(clos_n)

    def statistics(self):
        print("\n----------------")
        print("Total served:", self.served_job_count)
        print("Failed to allocate when gpus are enough:", len(self.allocation_failure_job))
        print("Distribution:", self.allocation_failure_job)
        print("----------------\n")
        util_rate = sum(self.utilization_rate_aggregated) + len(self.utilization_rate) / 1000.0 * np.mean(
            self.utilization_rate)
        count = len(self.utilization_rate_aggregated) + len(self.utilization_rate) / 1000.0
        util_rate = util_rate / count

        sg_rate = sum(self.spine_port_to_gpu_aggregated) + len(self.spine_port_to_gpu) / 1000.0 * np.mean(
            self.spine_port_to_gpu)
        count = len(self.spine_port_to_gpu_aggregated) + len(self.spine_port_to_gpu) / 1000.0
        sg_rate = sg_rate / count
        # print("sg1:", self.spine_port_to_gpu)
        # print("sg2:", self.spine_port_to_gpu_aggregated)
        return {"utilization": util_rate, "served": self.served_job_count, "spine_port_to_gpu_ratio": sg_rate}

    def final_statistics(self):
        block_diff = []
        block_diff_large = []

        fail_by_leaf = 0
        fail_by_spine = 0
        for e in self.allocation_failure_job_details:
            if len(e) < 6: continue
            if e[2] == "link_shortage":
                fail_by_leaf += 1
            else:
                fail_by_spine += 1
            print("\n----------------")
            print("GPU: %d/%d" % (e[0], np.sum(e[1])))
            block_diff.append(e[0] / e[-1])  # np.sum(e[1]))
            if e[0] >= 1024: block_diff_large.append(e[0] / e[-1])  # np.sum(e[1]))
            print("Cause:", e[2])
            gpu_s = [int(i) for i in e[1]]
            print("GPU status:", gpu_s)
            print("Leaf status:", e[4])
            print("Spine status:", e[3])
            print("----------------\n")

        print("\n\n************Total************")
        print("Total served:", self.served_job_count)
        print("Failed to allocate when gpus are enough:", len(self.allocation_failure_job))
        print("Distribution:", self.allocation_failure_job)
        print("Fail by leaf:%d  spine:%d" % (fail_by_leaf, fail_by_spine))
        print("Wait ratio:")
        keys = list(self.job_info.keys())
        keys.sort()
        for k in keys:
            print("%d: %f  %d/%d %f" % (
                k, self.job_info[k][1] / self.job_info[k][0], self.job_info[k][1], self.job_info[k][0],
                np.mean(self.job_info[k][2])))
        print("Avg diff when allocated:", np.mean(block_diff))
        print("Avg diff when allocated(large):", np.mean(block_diff_large))
        print("Diff 1024: ", np.mean(self.larger_allocaton))
        print("*****************************\n\n")


'''
        for i in range(self.chain_index):
            print("\n----------------",i)
            for e in self.allocation_chain[i]:
                print("GPU: %d/%d" % (e[0], np.sum(e[1])))
                print("Cause:", e[2])
                gpu_s = [int(i) for i in e[1]]
                print("GPU status:", gpu_s)
                print("Leaf status:", e[4])
                print("Spine status:", e[3])
            print("\n\n")
'''
