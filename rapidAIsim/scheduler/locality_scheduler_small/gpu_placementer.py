# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，在leaf_resource_manager中选取合适的leafgroup并更
#   新group信息,将相关信息传到server_manager中具体占用gpu并更新信息。这一过程
#   不涉及spine manager和connection manager。
# 2. 当需要跨leaf通信时，在server_resource_manager中按locality占用gpu并更新
#   gpu group，将gpu信息转化为leaf group的占用信息并在leaf_resource_manager中
#   更新leaf group的信息。同时选择spine交换机，此时有两种情况：
#   a. 若某几个跨leaf的gpu可以连到同一个spine下，那么当确定了leaf到spine的连接
#   关系后，调用整数规划求出oxc配置方案
#   b. 若需要spine迁移，从spine manager处得到迁移方案，传到gpu_placementer中，
#   gpu_placementer根据记录的job信息决定spine端口的迁移方案，即更新oxc_leaf_spine_map，
#   更新任务记录的相关信息，然后调用整数规划求出oxc配置方案

import copy
import math
import re
import time
from tabnanny import check
import rapidAIsim.scheduler.locality_scheduler_small.utils as utils
import rapidAIsim.scheduler.locality_scheduler_small.job as job
import rapidAIsim.scheduler.locality_scheduler_small.connection_manager as connection_manager
import rapidAIsim.scheduler.locality_scheduler_small.leaf_resource_manager as leaf_resource_manager
import rapidAIsim.scheduler.locality_scheduler_small.server_resource_manager as server_resource_manager
import rapidAIsim.scheduler.locality_scheduler_small.spine_resource_manager as spine_resource_manager


class GpuPlacementer4:
    def __init__(self,  spine_switch_num, leaf_switch_num, spine_switch_port_num, leaf_switch_port_num, server_num, oxc_num = 32):
        self.gpu_num = spine_switch_num*spine_switch_port_num
        self.server_num = server_num
        self.leaf_num = leaf_switch_num
        self.spine_num = spine_switch_num
        self.oxc_num = oxc_num
        self.gpu_per_server = int(self.gpu_num/server_num)
        self.gpu_per_leaf = int(self.gpu_num/self.leaf_num)
        self.port_per_spine = int(self.gpu_num/self.spine_num)
        print("Cluster Info:")
        print("server_num: "+" ")
        print(server_num)
        print("leaf_num: "+" ")
        print(leaf_switch_num)
        print("gpu_num: "+" ")
        print(self.gpu_num)
        self.server_resource_manager_ = server_resource_manager.ServerResourceManager(server_num, self.gpu_per_server, self.leaf_num)
        self.leaf_resource_manager_ = leaf_resource_manager.LeafResourceManager(self.leaf_num, self.gpu_per_leaf)
        self.spine_resource_manager_ = spine_resource_manager.SpineSwitchManager(self.spine_num, self.port_per_spine, [])
        self.connection_manager_ = connection_manager.ConnectionManager(self.gpu_num, self.server_num, self.leaf_num, self.spine_num, self.oxc_num)

        # job queue
        self.current_job_list = {}
        self.history_job_list = {}

    def schedule(self, gpu_num, job_id, sim_time, queued_jobs, spine_strategy_mode ="gpu_first", oxc_need_reconfig = False):
        from rapidAIsim.core.simulator import Simulator
        print("some job arrive: "+str(job_id)+","+str(gpu_num))
        time_start = time.perf_counter()
        new_job = job.Job(job_id)
        chosen_gpu_list = []
        allocation_link_mapping = []
        # 情况零：GPU数量不足
        if gpu_num > self.server_resource_manager_.cal_remain_gpu_num():
            print("finish allocation, no resource due to GPU")
            return False, None, None, None,None,None,None

        # 否则需要跨leaf通信
        choose_group_in_spine_result = self.spine_resource_manager_.choose_group_in_spine(gpu_num)
        if(choose_group_in_spine_result[0] and len(choose_group_in_spine_result[1]) ==1 ):
        # 情况二：跨leaf通信但不需要spine迁移
            choosed_spine_index_list = choose_group_in_spine_result[1]
            # 如果存在符合条件的leaf/spine，就占用
            leaf_remain_empt_server_list = []
            for temp_leaf_id in range(self.leaf_num):
                leaf_remain_empt_server_list.append(0)
            for temp_server_id in range(self.server_num):
                temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
                if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
                    leaf_remain_empt_server_list[temp_leaf_id] += 1
            chosen_gpu_list = []
            job_allocated_oxc_spine_link = {}
            job_used_spine_port_num_pair = {}
            for chosen_spine_id in choosed_spine_index_list:
                chosen_group_size = int(gpu_num/len(choosed_spine_index_list))
                valid, server_occupy_gpuNum_map = self.connection_manager_.find_valid_gpu_for_specific_spine(chosen_group_size, chosen_spine_id, self.server_resource_manager_.return_server_remain_gpuNum_map(),job_allocated_oxc_spine_link,job_used_spine_port_num_pair, leaf_remain_empt_server_list)
                if(not valid):
                    self.spine_resource_manager_.release_spine_group_with_give_id_and_group(chosen_spine_id, chosen_group_size)
                    print("finish allocation, no resource due to locality3")
                    return False, None, None, None,None,None,None
                for server_id in server_occupy_gpuNum_map:
                    if server_occupy_gpuNum_map[server_id]>0:
                        chosen_gpu_list.extend(self.server_resource_manager_.server_list[server_id].occupy_gpu_with_required_num(server_occupy_gpuNum_map[server_id])[1])
            chosen_leaf_id_num_list = self.leaf_resource_manager_.update_group_with_given_gpu_list(chosen_gpu_list)

            temp_leaf_to_spine_map = {} # key 为leaf的index，value为另一个map B， map B的key为spine交换机的index，value为该leaf要新连多少根线到该spine
            for choosed_leaf_id_num_pair in chosen_leaf_id_num_list:
                temp_leaf_to_each_spine_map = {}
                for choosed_spine_index in choosed_spine_index_list:
                    temp_leaf_to_each_spine_map[choosed_spine_index] = int(choosed_leaf_id_num_pair[1]/len(choosed_spine_index_list))
                temp_leaf_to_spine_map[choosed_leaf_id_num_pair[0]] = temp_leaf_to_each_spine_map

            new_job.start_time = sim_time
            new_job.allocated_gpus = chosen_gpu_list
            new_job.job_leaf_to_spine_map = temp_leaf_to_spine_map
            new_job.allocated_oxc_spine_link = job_allocated_oxc_spine_link
            new_job.used_spine_port_num_pair = job_used_spine_port_num_pair
            self.current_job_list[job_id] = new_job
            allocation_link_mapping,record_leaf_num_map,record_spine_num_map = self.translate_updated_links(chosen_gpu_list, job_allocated_oxc_spine_link)
            print("finish allocation assign whole clos for large job")
            self.check_spine()
            f2 = open('queue_length.txt','a')
            f2.write(str(len(queued_jobs)))
            f2.write(",")
            f2.write(str(sim_time) )
            f2.write("\n" )
            f2.close()     
            if len(chosen_gpu_list) != gpu_num:
                print(len(chosen_gpu_list), gpu_num, len(choosed_spine_index_list))
            assert len(chosen_gpu_list) == gpu_num
            time_end = time.perf_counter()
            time_sum = time_end-time_start
            
            Simulator.SCHEDULER_TIME_COST[job_id] = 0
            f3 = open('schedule_time_cost.txt','a')
            f3.write(str(job_id))
            f3.write(",")
            f3.write(str(time_sum) )
            f3.write("\n" )
            f3.close()  
            return True, True, chosen_gpu_list, allocation_link_mapping,None,None,None
        # 情况三：跨leaf通信且需要spine迁移
        else:
            self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
            self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
            # print("no resource3 start migration")
            # self.do_spine_migration()
            # return self.schedule(gpu_num, job_id, sim_time, queued_jobs)
            return False, None, None, None,None,None,None
           
    def do_spine_migration(self):
        self.check_spine()
        self.spine_resource_manager_.print_remain_spoine_port_num()
        self.spine_resource_manager_.clear_spine_list()
        self.connection_manager_.clear_spine_and_oxc()
        temp_size = 0
        for temp_job_key in self.current_job_list:
            temp_job = self.current_job_list[temp_job_key]
            new_temp_size = 0
            for chosen_spine_id in temp_job.used_spine_port_num_pair:
                new_temp_size+=temp_job.used_spine_port_num_pair[chosen_spine_id]
            if(new_temp_size):
                temp_gpu_reqiure_num = len(temp_job.allocated_gpus)
                choose_group_in_spine_result = self.spine_resource_manager_.choose_group_in_spine(temp_gpu_reqiure_num)
                assert(choose_group_in_spine_result[0])
                job_used_spine_port_num_pair = {}
                choosed_spine_index_list = choose_group_in_spine_result[1]
                for chosen_spine_id in choosed_spine_index_list:
                    chosen_group_size = int(temp_gpu_reqiure_num/len(choosed_spine_index_list))
                    job_used_spine_port_num_pair[chosen_spine_id] =  chosen_group_size
                    temp_size+=chosen_group_size
                # 记录原来占用的leaf资源
                temp_chosen_leaf_id_num_list = []
                leaf_num_map = {}
                for leaf_id in temp_job.job_leaf_to_spine_map:
                    for spine_id in  temp_job.job_leaf_to_spine_map[leaf_id]:
                        if(temp_job.job_leaf_to_spine_map[leaf_id][spine_id]>0):
                            if leaf_id not in leaf_num_map:
                                leaf_num_map[leaf_id] = 0
                            leaf_num_map[leaf_id]+=temp_job.job_leaf_to_spine_map[leaf_id][spine_id]
                for item in leaf_num_map:
                    temp_chosen_leaf_id_num_list.append([item, leaf_num_map[item]])
                temp_oxc_leaf_spine_map , temp_leaf_to_spine_map, job_allocated_oxc_spine_link = self.connection_manager_.update_leaf_to_spine_map_according_to_chosen_leaf_and_spine(temp_chosen_leaf_id_num_list, choosed_spine_index_list)
                temp_job.job_leaf_to_spine_map = temp_leaf_to_spine_map
                temp_job.allocated_oxc_spine_link = job_allocated_oxc_spine_link
                temp_job.used_spine_port_num_pair = job_used_spine_port_num_pair
        print("finish migration", temp_size, self.spine_resource_manager_.cal_remain_spoine_port_num())
        self.spine_resource_manager_.print_remain_spoine_port_num()


                
    def update_finished_job(self, job_id, sim_time, queued_jobs):
        print("some job finish" + str(job_id))
        to_leave_job = copy.deepcopy(self.current_job_list[job_id])
        to_leave_job.finish_time = sim_time
        self.history_job_list[job_id] = to_leave_job
        self.server_resource_manager_.release_gpu_in_server(to_leave_job.allocated_gpus)
        self.leaf_resource_manager_.release_group_with_given_gpu_list(to_leave_job.allocated_gpus)
        spine_portNum_map = {}
        for oxc_id in to_leave_job.allocated_oxc_spine_link:
            for leaf_id in to_leave_job.allocated_oxc_spine_link[oxc_id]:
                spine_id = to_leave_job.allocated_oxc_spine_link[oxc_id][leaf_id]
                if spine_id not in spine_portNum_map:
                    spine_portNum_map[spine_id] = 0
                spine_portNum_map[spine_id] += 1
        for spine_id in spine_portNum_map:
            self.spine_resource_manager_.release_spine_group_with_give_id_and_group(spine_id, spine_portNum_map[spine_id])
        self.connection_manager_.release_connection_resource(to_leave_job.allocated_oxc_spine_link)
        del self.current_job_list[job_id]
        f2 = open('queue_length.txt','a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()        
        self.check_spine()
    
    def check_spine(self):
        temp_size = 0
        for temp_job_key in self.current_job_list:
            temp_job = self.current_job_list[temp_job_key]
            for chosen_spine_id in temp_job.used_spine_port_num_pair:
                temp_size+=temp_job.used_spine_port_num_pair[chosen_spine_id]
        # print(self.gpu_num-temp_size, self.spine_resource_manager_.cal_remain_spoine_port_num())
        assert(self.gpu_num-temp_size==self.spine_resource_manager_.cal_remain_spoine_port_num())
        

    def translate_updated_links(self, gpu_indexes, updated_links):
        record_leaf_num_map = {}
        record_spine_num_map = {}
        allocation_link_mapping = []
        # gpu - leaf links
        for output_gpu_index in gpu_indexes:
            output_leaf_index = utils.get_leaf_module_id(int(output_gpu_index/self.gpu_per_leaf), self.gpu_num)
            allocation_link_mapping.append((output_gpu_index, output_leaf_index, 1))
            allocation_link_mapping.append((output_leaf_index, output_gpu_index, 1))
        # leaf - spine links
        temp_leaf_to_spine_num = {}
        for oxc_id in updated_links:
            for leaf_id in updated_links[oxc_id]:
                spine_id = updated_links[oxc_id][leaf_id]
                if (leaf_id,spine_id) not in temp_leaf_to_spine_num:
                    temp_leaf_to_spine_num[(leaf_id,spine_id)] = 0
                temp_leaf_to_spine_num[(leaf_id,spine_id)] += 1

                if leaf_id not in record_leaf_num_map:
                    record_leaf_num_map[leaf_id] = 0
                record_leaf_num_map[leaf_id] += 1
                if spine_id not in record_spine_num_map:
                    record_spine_num_map[spine_id] = 0
                record_spine_num_map[spine_id] += 1
        for leaf_spine_pair in temp_leaf_to_spine_num:
            allocation_link_mapping.append((leaf_spine_pair[0]+self.gpu_num, leaf_spine_pair[1]+self.gpu_num+self.leaf_num, temp_leaf_to_spine_num[leaf_spine_pair]))
            allocation_link_mapping.append((leaf_spine_pair[1]+self.gpu_num+self.leaf_num,leaf_spine_pair[0]+self.gpu_num, temp_leaf_to_spine_num[leaf_spine_pair]))
        return allocation_link_mapping,record_leaf_num_map,record_spine_num_map


            
            