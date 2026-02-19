from math import ceil
import queue
import math
import time
import logging
import random

import rapidAIsim.scheduler.static_scheduler.job as job
import rapidAIsim.scheduler.static_scheduler.switch_resource as switch
from rapidAIsim.scheduler.static_scheduler.utils import *
from rapidAIsim.scheduler.static_scheduler.job_request import JobRequest


logging.basicConfig(level= logging.ERROR)

class static_scheduler:
    def __init__(self, clos_n, clos_m,  server_num = 4):#spine_switch_num, leaf_switch_num, spine_switch_port_num, leaf_switch_port_num):
        spine_switch_num = clos_n
        leaf_switch_num = clos_m
        spine_switch_port_num = 64
        leaf_switch_port_num = 64


        self.spine_num = spine_switch_num
        self.leaf_num = leaf_switch_num
        self.gpu_per_switch = int(leaf_switch_port_num / 2)
        self.gpu_num = self.gpu_per_switch * leaf_switch_num
        self.spine_switch_port_num = spine_switch_port_num
        self.leaf_switch_port_num = leaf_switch_port_num
        self.gpu_per_server = int(self.gpu_num/server_num)
        self.gpu_per_leaf = int(self.gpu_num/self.leaf_num)
        self.server_per_leaf = int(self.gpu_per_leaf/self.gpu_per_server)
        self.port_per_spine = int(self.gpu_num/self.spine_num)

        self.gpus = []
        for i in range(self.leaf_num):
            self.gpus.append(np.zeros(int(leaf_switch_port_num / 2)))
        for i in range(self.leaf_num):
            for j in range(int(leaf_switch_port_num / 2)):
                if j%2 == 1:
                    self.gpus[i][j] = 1

        logging.debug("cluster size: %d gpus" % self.gpu_num)
        #assert spine_switch_port_num == self.leaf_num and int(leaf_switch_port_num/2) == self.spine_num and self.gpu_num == int(self.spine_num * spine_switch_port_num)

        # switches
        self.spine_switches = []
        self.leaf_switches = []
        for i in range(self.spine_num):
            s = switch.switch(spine_switch_port_num)
            self.spine_switches.append(s)
        for i in range(self.leaf_num):
            s = switch.switch(leaf_switch_port_num)
            s.take_up_port_by_num(int(leaf_switch_port_num / 2)) # half of the links are downlinks taken by gpu
            self.leaf_switches.append(s)

        # job queue
        self.reqQueue = queue.Queue()
        self.next_req = None

        # allocation method
        self.allocate_methods = ["static_based"]
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


        # thresh-based policy adjustment
        self.thresh_ratio = 1.0
        self.thresh_line = int(self.thresh_ratio * spine_switch_port_num)
        self.even_allocation_num = 0  # the number of even allocation to all spine servers
        self.to_decrease_thresh = False

        # optimization time limit
        self.time_limit = -1  # reserved

        # conflict detection
        self.failed_by_conflict = False
        self.conflict_time_point = -1
        self.current_running_task = 0
        self.current_gpu_num = 0
        self.gpu_class_num_map = {0 for i in range(int(math.log2(spine_switch_num*spine_switch_port_num)))}

        self.leaf_port_list = [[0 for i in range(self.gpu_per_leaf)] for i in range(self.leaf_num)]
        self.single_spine_port_list = [0 for i in range(self.port_per_spine)]
        self.job_gpu_spine_port_map = {}
        self.job_gpu_leaf_port_map = {}

    def choose_a_leaf_port(self, leaf_id):
        to_chose_port_id = -1
        to_chose_port_id = random.randint(0,self.gpu_per_leaf-1)
        self.leaf_port_list[leaf_id][to_chose_port_id] += 1
        return to_chose_port_id

    def choose_a_spine_port(self, port_id):
        # assert self.single_spine_port_list[port_id] == 0
        self.single_spine_port_list[port_id] += 1


    def choose_leaf_spine_port(self, job_id, choose_leaf_port_map, choose_spine_port_num, gpu_list):
        to_choose_leaf_port_global = []
        for leaf_id in choose_leaf_port_map:
            for temp_chosen_num in range(choose_leaf_port_map[leaf_id]):
                to_choose_leaf_port_global.append(leaf_id*self.gpu_per_leaf + self.choose_a_leaf_port(leaf_id) + self.gpu_num)
        to_choose_spine_port_global = []
        for temp_chosen_num in range(choose_spine_port_num):
            to_choose_spine_port_global.append(to_choose_leaf_port_global[temp_chosen_num]+self.gpu_num)
            self.choose_a_spine_port(to_choose_leaf_port_global[temp_chosen_num]-self.gpu_num)
        assert len(to_choose_leaf_port_global) == len(to_choose_spine_port_global)
        assert len(to_choose_leaf_port_global) == len(gpu_list)

        temp_leaf_spine_pair = []
        for temp_id in range(len(to_choose_leaf_port_global)):
            temp_leaf_spine_pair.append((gpu_list[temp_id], to_choose_spine_port_global[temp_id]))
        self.job_gpu_spine_port_map[job_id] = temp_leaf_spine_pair

        temp_gpu_leaf_pair = []
        for temp_id in range(len(to_choose_leaf_port_global)):
            temp_gpu_leaf_pair.append((gpu_list[temp_id], to_choose_leaf_port_global[temp_id]))
        self.job_gpu_leaf_port_map[job_id] = temp_gpu_leaf_pair

    def relase_leaf_spine_port(self, job_id):
        temp_gpu_spine_pair = self.job_gpu_spine_port_map[job_id]
        for gpu_spine_port_pair in temp_gpu_spine_pair:
            #assert self.single_spine_port_list[gpu_spine_port_pair[1]-2*self.gpu_num] == 1
            self.single_spine_port_list[gpu_spine_port_pair[1]-2*self.gpu_num] -= 1

        temp_gpu_leaf_pair = self.job_gpu_leaf_port_map[job_id]
        for gpu_leaf_port_pair in temp_gpu_leaf_pair:
            leaf_id = int((gpu_leaf_port_pair[1]-self.gpu_num)/self.gpu_per_leaf)
            leaf_loacl_port = (gpu_leaf_port_pair[1]-self.gpu_num) - leaf_id*self.gpu_per_leaf
            self.leaf_port_list[leaf_id][leaf_loacl_port] -= 1

        del self.job_gpu_spine_port_map[job_id]
        del self.job_gpu_leaf_port_map[job_id]

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

    def schedule(self, gpu_num, job_id, sim_time, queued_jobs=[]):
        temp_z = pow(2,int(math.log2(int(gpu_num))))
        time_start = time.perf_counter()
        next_req = JobRequest(gpu_num, job_id)

        # debug
        self.should_schedule = False
        self.wasted_time += (sim_time - self.last_update_time)

        # current gpu use rate
        self.used_gpu_num = np.sum(self.gpus)
        

        allocate_success, cause_of_failure, algo_gpu_allocation, algo_spine_allocation = self.allocate_GPU(next_req, sim_time, [2,4])
        if allocate_success:
            all_gpu_index,link_mapping = None, None
            gpu_indexes = self._translate_gpu_index(algo_gpu_allocation)
            self.utilization_rate.append(self.used_gpu_num / float(self.gpu_num))
            self.utilization_time.append(sim_time - self.time_slot)
            self.time_slot = sim_time
            self.current_gpu_num += gpu_num
            f2 = open('queue_length.txt','a')
            f2.write(str(len(queued_jobs)))
            f2.write(",")
            f2.write(str(sim_time) )
            f2.write("\n" )
            f2.close()
            choose_leaf_port_map = {}
            for gpu_id in range(len(gpu_indexes)):
                leaf_id = int(gpu_id/self.gpu_per_leaf)
                if leaf_id not in choose_leaf_port_map:
                    choose_leaf_port_map[leaf_id] = 0
                choose_leaf_port_map[leaf_id] += 1
            if len(gpu_indexes)>self.gpu_per_server: #RTODO
                self.choose_leaf_spine_port(job_id, choose_leaf_port_map, len(gpu_indexes), gpu_indexes)
            else:
                self.job_gpu_leaf_port_map[job_id] = {}
                self.job_gpu_spine_port_map[job_id] = {}
        else:
            gpu_indexes = None
        if gpu_indexes != None:
            pow_2_gpu_list = gpu_indexes[:temp_z]
            remain_gpu_list = gpu_indexes[temp_z:]
        # print(self.job_gpu_leaf_port_map)
        if allocate_success:
            return allocate_success, gpu_indexes, self.job_gpu_leaf_port_map[job_id], self.job_gpu_spine_port_map[job_id] #allocate_success, gpu_indexes, link_mapping
        else:
            return allocate_success, gpu_indexes, None, None

    def update_finished_job(self, job_id, sim_time, queued_jobs=[]):
        move_flag = False
        if job_id in self.job_gpu_spine_port_map:
            self.relase_leaf_spine_port(job_id)
        if not self.should_schedule:
            self.last_update_time = sim_time
            self.should_schedule = True

        # current gpu use rate
        self.used_gpu_num = np.sum(self.gpus)
        job_gpu_num = 0
        for i, ongoing_job in enumerate(self.allocated_jobs):
            if ongoing_job.id == job_id:
                # release resources
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

                self.utilization_rate.append(self.used_gpu_num / float(self.gpu_num))
                self.utilization_time.append(sim_time - self.time_slot)
                self.time_slot = sim_time
                break
        self.current_gpu_num -= job_gpu_num
        f2 = open('queue_length.txt','a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()
        if not move_flag:
            logging.error("Update status failed! : invalid job_id")
            exit()

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

            return False


    # if leaf switches have enough uplinks
    def check_leaf_allocation(self, leaf_switch_set, clos_n):
        link_shortage = []
        if len(leaf_switch_set) < 2: return link_shortage
        for li in leaf_switch_set:
            if self.leaf_switches[li].free_port_num < clos_n:
                link_shortage.append(li)
        # assert  len(link_shortage) == 0
        return link_shortage


    

    def allocate_GPU(self, next_req: JobRequest, sim_time, banned_server_list = [2,4]):
        #### cause of allocation failure
        failure_cause = ""
        self.failed_by_conflict = False


        #self.allocate_method = "arbitrary"

        if self.allocate_method == "static_based":
            # a naive way for allocating GPUs as split clos
            if (self.gpu_num - np.sum(self.gpus)) < next_req.gpu_num:
                print("no resource0", self.gpu_num - np.sum(self.gpus), next_req.gpu_num)
                return False, failure_cause,None, None

            # decide gpu allocation
            free_gpus = -np.sum(self.gpus, axis=1) + self.gpu_per_switch
            remain_ports = [(self.gpu_per_switch - l.free_port_num) for l in self.leaf_switches]
            request_num = next_req.gpu_num

            allocation_success = False
            gpu_indexes = []
            clos_n = -1
            clos_m = -1
            ss_indexes = None
            gpu_indexes = []
            # simply see if there are enough remaining GPU
            if (self.gpu_num - np.sum(self.gpus) -8) < next_req.gpu_num:
                return False, failure_cause, None, None
            else:
                distrib_count = 0
                new_job = job.job(next_req.request_id)  # next_req.exec_time)
                
                require_server_num = ceil(request_num/self.gpu_per_server)
                temp_server_remain_gpu_map = {}
                for li in range(self.leaf_num):
                    for j in range(self.gpu_per_switch):  # we do not care about the gpu
                        temp_server_id = int(j/self.gpu_per_server)+self.server_per_leaf*li
                        if temp_server_id not in banned_server_list:
                            temp_gpu_id = j+self.gpu_per_switch*li
                            if self.gpus[li][j] == 0:
                                if temp_server_id not in temp_server_remain_gpu_map:
                                    temp_server_remain_gpu_map[temp_server_id] = []
                                temp_server_remain_gpu_map[temp_server_id].append(temp_gpu_id)
                temp_server_remain_gpu_map = dict( sorted(temp_server_remain_gpu_map.items(),key = lambda x:len(x[1]),reverse = False))
                temp_server_remain_gpu_list = []
                for server_id in temp_server_remain_gpu_map:
                    temp_remain_gpu_list = temp_server_remain_gpu_map[server_id]
                    if len(temp_remain_gpu_list)==self.gpu_per_server or len(temp_remain_gpu_list)>=next_req.gpu_num:
                        temp_server_remain_gpu_list.append([server_id, temp_remain_gpu_list])
                if len(temp_server_remain_gpu_list)<require_server_num:
                    return False, failure_cause, None, None
                else:
                    for i in range(require_server_num):
                        for gpu_id in temp_server_remain_gpu_list[i][1]:
                            if distrib_count< next_req.gpu_num:
                                distrib_count+=1
                                gpu_indexes.append((int(gpu_id/self.gpu_per_switch), gpu_id%self.gpu_per_switch))
                                new_job.add_gpu((int(gpu_id/self.gpu_per_switch), gpu_id%self.gpu_per_switch))
                    link_conflicted = False
                    # if check_conflict: link_conflicted = self.check_conflict(gpu_indexes)
                    if not link_conflicted:
                        self.allocated_jobs.append(new_job)
                        for pair in gpu_indexes:
                            self.gpus[pair[0]][pair[1]] = 1
                        # print("gpu_indexes:",gpu_indexes)
                        return True, failure_cause, gpu_indexes, None
                    else:
                        return False, failure_cause, None, None

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


# scheduler = StaticScheduler(1, 2, 8)
# success,gpu_index,gpu_leaf,gpu_spine = scheduler.schedule(4,0,0)
# print(gpu_index)
# #print(gpu_leaf,gpu_spine)
# # print(scheduler.leaf_port_list)
# # print(scheduler.single_spine_port_list)
# print(np.sum(scheduler.gpus))
# success,gpu_index,gpu_leaf,gpu_spine = scheduler.schedule(4,1,1)
# print(gpu_index)
# #print(gpu_leaf,gpu_spine)
# # print(scheduler.leaf_port_list)
# # print(scheduler.single_spine_port_list)
# print(np.sum(scheduler.gpus))
# success,gpu_index,gpu_leaf,gpu_spine = scheduler.schedule(2,2,1)
# print(gpu_index)
# #print(gpu_leaf,gpu_spine)
# # print(scheduler.leaf_port_list)
# # print(scheduler.single_spine_port_list)
# print(np.sum(scheduler.gpus))
# scheduler.update_finished_job(1,4)
# # print(scheduler.leaf_port_list)
# # print(scheduler.single_spine_port_list)
# print(np.sum(scheduler.gpus))
# scheduler.update_finished_job(2,4)
# # print(scheduler.leaf_port_list)
# # print(scheduler.single_spine_port_list)
# print(np.sum(scheduler.gpus))
# scheduler.update_finished_job(0,4)
# # print(scheduler.leaf_port_list)
# # print(scheduler.single_spine_port_list)
# print(np.sum(scheduler.gpus))

