import copy
import math
import re
import time
from tabnanny import check
import rapidAIsim.scheduler.locality_scheduler.utils as utils
import rapidAIsim.scheduler.locality_scheduler.job as job
import rapidAIsim.scheduler.locality_scheduler.connection_manager as connection_manager
import rapidAIsim.scheduler.locality_scheduler.leaf_resource_manager as leaf_resource_manager
import rapidAIsim.scheduler.locality_scheduler.server_resource_manager as server_resource_manager
import rapidAIsim.scheduler.locality_scheduler.spine_resource_manager as spine_resource_manager


def twopart(n):
    return n&(n-1) == 0


class GpuPlacementer:
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
        # from rapidAIsim.core.simulator import Simulator
        # if 'banned_spine_num' in Simulator.CONF_DICT:
        #     ban_spine_num = int(Simulator.CONF_DICT['banned_spine_num'])
        #     self.spine_resource_manager_ = spine_resource_manager.SpineSwitchManager(self.spine_num, self.port_per_spine, [i for i in range(ban_spine_num)])
        #     print("debug [i for i in range(banned_spine_num)]", [i for i in range(ban_spine_num)])
        # else:
        #     self.spine_resource_manager_ = spine_resource_manager.SpineSwitchManager(self.spine_num, self.port_per_spine, [])
        #     print("debug no banned_spine_num")
        #     print(Simulator.CONF_DICT)
        self.spine_resource_manager_ = spine_resource_manager.SpineSwitchManager(self.spine_num, self.port_per_spine, [])
        self.connection_manager_ = connection_manager.ConnectionManager(self.gpu_num, self.server_num, self.leaf_num, self.spine_num, self.oxc_num)

        # job queue
        self.current_job_list = {}
        self.history_job_list = {}
        
    def fusion_gpu_list(self, pow_2_gpu_list, remain_gpu_list):
        communication_pair_list = []
        fus_gpu_list = []
        fus_gpu_list.extend(pow_2_gpu_list)
        fus_gpu_list.extend(remain_gpu_list)
        gpu_global_local_index_map = {}
        for i in range(len(fus_gpu_list)):
            gpu_global_local_index_map[fus_gpu_list[i]] = i
        leaf_pow_2_gpu_list_map = {}
        for pow_2_gpu in pow_2_gpu_list:
            leaf_id = int(pow_2_gpu/self.gpu_per_leaf)
            if leaf_id not in leaf_pow_2_gpu_list_map:
                leaf_pow_2_gpu_list_map[leaf_id] = []
            leaf_pow_2_gpu_list_map[leaf_id].append(pow_2_gpu)
        leaf_remain_gpu_list_map = {}
        for remain_gpu in remain_gpu_list:
            leaf_id = int(remain_gpu/self.gpu_per_leaf)
            if leaf_id not in leaf_remain_gpu_list_map:
                leaf_remain_gpu_list_map[leaf_id] = []
            leaf_remain_gpu_list_map[leaf_id].append(remain_gpu)
        for leaf_id in leaf_remain_gpu_list_map:
            for remain_gpu in leaf_remain_gpu_list_map[leaf_id]:
                to_comm_gpu = leaf_pow_2_gpu_list_map[leaf_id][0]
                communication_pair_list.append((gpu_global_local_index_map[remain_gpu], gpu_global_local_index_map[to_comm_gpu]))
                del(leaf_pow_2_gpu_list_map[leaf_id][0])
        return communication_pair_list

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
        if not self.server_resource_manager_.whether_can_find_valid_server(gpu_num):
            print("finish allocation, no resource due to locality")
            return False, None, None, None,None,None,None
        # 情况一：尝试不跨leaf通信 
        print("Stage 1")
        potentional_leaf_list = []
        # Step1. 在leaf_resource_manager中选取合适的leafgroup
        for temp_leaf_id in range(self.leaf_num):
            require_server_num = math.ceil(gpu_num/self.gpu_per_server)
            require_gpu_num_in_server = min(self.gpu_per_server,gpu_num)
            valid_server_num = 0
            for temp_server_num in range(int(temp_leaf_id*self.gpu_per_leaf/self.gpu_per_server), int((1+temp_leaf_id)*self.gpu_per_leaf/self.gpu_per_server),1):
                if self.server_resource_manager_.server_list[temp_server_num].remain_gpu_num()>=require_gpu_num_in_server:
                    valid_server_num += 1
            if valid_server_num>=require_server_num:
                potentional_leaf_list.append([temp_leaf_id, sum(self.leaf_resource_manager_.leaf_list[temp_leaf_id].leaf_group)])
        potentional_leaf_list.sort( key=lambda x: (x[1])) 
        if len(potentional_leaf_list)>0:
            temp_leaf_id = potentional_leaf_list[0][0]
            #  Step2 在选择的leaf交换机下联的server中按照locality选择gpu
            chosen_gpu_list = self.server_resource_manager_.choose_gpu_in_one_leaf(temp_leaf_id, gpu_num)
            self.leaf_resource_manager_.leaf_list[temp_leaf_id].update_leaf_group_with_required_num(gpu_num)
            # gpu - leaf links
            for output_gpu_index in chosen_gpu_list:
                assert int(output_gpu_index/self.gpu_per_leaf) == temp_leaf_id
                output_leaf_index = utils.get_leaf_module_id(temp_leaf_id, self.gpu_num)
                allocation_link_mapping.append([output_gpu_index, output_leaf_index, 1])
                allocation_link_mapping.append([output_leaf_index, output_gpu_index, 1])
            #  记录job
            new_job.start_time = sim_time
            new_job.allocated_gpus = chosen_gpu_list
            self.current_job_list[job_id] = new_job
            print("finish allocation one leaf")
            self.check_spine()
            f2 = open('queue_length.txt','a')
            f2.write(str(len(queued_jobs)))
            f2.write(",")
            f2.write(str(sim_time) )
            f2.write("\n" )
            f2.close()
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
        # 否则需要跨leaf通信

        # 尝试不用spine交换机
        print("start trying leaf connection directly")
        leaf_remain_empt_server_list = []
        for temp_leaf_id in range(self.leaf_num):
            leaf_remain_empt_server_list.append(0)
        for temp_server_id in range(self.server_num):
            temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
            if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
                leaf_remain_empt_server_list[temp_leaf_id] += 1
        tmp_allocate_success, leaf_to_used = self.connection_manager_.find_valid_leaf_pair(gpu_num, leaf_remain_empt_server_list)
        if tmp_allocate_success:
            chosen_gpu_list = []
            for tmp_leaf_id in leaf_to_used:
                tmp_chosen_gpu_list = self.server_resource_manager_.choose_gpu_in_one_leaf(tmp_leaf_id, int(gpu_num/2))
                self.leaf_resource_manager_.leaf_list[tmp_leaf_id].update_leaf_group_with_required_num(int(gpu_num/2))
                chosen_gpu_list.extend(tmp_chosen_gpu_list)
                # gpu - leaf links
                for output_gpu_index in tmp_chosen_gpu_list:
                    assert int(output_gpu_index/self.gpu_per_leaf) == tmp_leaf_id
                    output_leaf_index = utils.get_leaf_module_id(tmp_leaf_id, self.gpu_num)
                    allocation_link_mapping.append([output_gpu_index, output_leaf_index, 1])
                    allocation_link_mapping.append([output_leaf_index, output_gpu_index, 1])
            # leaf - leaf links
            assert len(leaf_to_used) == 2
            tmp_leaf_1_index = leaf_to_used[0] + self.gpu_num
            tmp_leaf_2_index = leaf_to_used[1] + self.gpu_num
            allocation_link_mapping.append([tmp_leaf_1_index, tmp_leaf_2_index, int(gpu_num/2)])
            allocation_link_mapping.append([tmp_leaf_2_index, tmp_leaf_1_index, int(gpu_num/2)])
            #  记录job
            new_job.start_time = sim_time
            new_job.allocated_gpus = chosen_gpu_list
            self.current_job_list[job_id] = new_job
            print("finish allocation one leaf")
            self.check_spine()
            f2 = open('queue_length.txt','a')
            f2.write(str(len(queued_jobs)))
            f2.write(",")
            f2.write(str(sim_time) )
            f2.write("\n" )
            f2.close()
            time_end = time.perf_counter()
            time_sum = time_end-time_start
            
            Simulator.SCHEDULER_TIME_COST[job_id] = 0
            f3 = open('schedule_time_cost.txt','a')
            f3.write(str(job_id))
            f3.write(",")
            f3.write(str(time_sum) )
            f3.write("\n" )
            f3.close()  
            print("successfully leaf connection directly")
            return True, True, chosen_gpu_list, allocation_link_mapping,None,None,None


        # 尝试用spine交换机
        choose_group_in_spine_result = self.spine_resource_manager_.choose_group_in_spine(gpu_num)
        if(choose_group_in_spine_result[0] and len(choose_group_in_spine_result[1])==1) and  twopart(gpu_num):
            # 尽可能在同一个spine
            print("stage 2")
            choosed_spine_index_list = choose_group_in_spine_result[1]
            print("debug choosed_spine_index_list",choosed_spine_index_list)
            self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
            self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
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
            temp_spine_index = 0
            for chosen_spine_id in choosed_spine_index_list:
                chosen_group_size = choose_group_in_spine_result[2][temp_spine_index]
                temp_spine_index += 1
                if chosen_group_size != int(gpu_num/len(choosed_spine_index_list)):
                    print("fuck", choose_group_in_spine_result[2], gpu_num)
                valid, server_occupy_gpuNum_map = self.connection_manager_.find_valid_gpu_for_specific_spine(chosen_group_size, chosen_spine_id, self.server_resource_manager_.return_server_remain_gpuNum_map(),job_allocated_oxc_spine_link,job_used_spine_port_num_pair, leaf_remain_empt_server_list)
                if(not valid):
                    self.spine_resource_manager_.release_spine_group_with_give_id_and_group(chosen_spine_id, chosen_group_size)
                    print("finish allocation, no resource due to locality3", len(choosed_spine_index_list), gpu_num)
                    self.leaf_resource_manager_.print_remain_leaf_port_num()
                    self.spine_resource_manager_.print_remain_spoine_port_num()
                    self.spine_resource_manager_.print_resource_info()
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
        elif choose_group_in_spine_result[0] and len(choose_group_in_spine_result[1])>1 and twopart(gpu_num):
            print("start whole leaf-spine")
            self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
            self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
            job_allocated_oxc_spine_link = {}
            job_used_spine_port_num_pair = {}

            choosed_spine_index_list = choose_group_in_spine_result[1]
            require_leaf_num = int(gpu_num/self.gpu_per_leaf) #TODO
            chosen_leaf_id_num_list = []
            for leaf_spine in self.leaf_resource_manager_.leaf_list:
                if self.gpu_per_leaf in leaf_spine.leaf_group and len(chosen_leaf_id_num_list)<require_leaf_num:
                    chosen_leaf_id_num_list.append([leaf_spine.leaf_id,self.gpu_per_leaf])
            if len(chosen_leaf_id_num_list)>=require_leaf_num:
                print(print("stage 2.2"))
                chosen_gpu_list = []
                for chosen_leaf_id_num in chosen_leaf_id_num_list:
                    chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf(chosen_leaf_id_num[0], self.gpu_per_leaf))
                    self.leaf_resource_manager_.leaf_list[chosen_leaf_id_num[0]].update_leaf_group_with_required_num(self.gpu_per_leaf)
                print("debug chosen_leaf_id_num_list", chosen_leaf_id_num_list)
                print("debug choosed_spine_index_list", choosed_spine_index_list)
                print("debug job_allocated_oxc_spine_link", job_allocated_oxc_spine_link)
                print("debug job_used_spine_port_num_pair", job_used_spine_port_num_pair)
                temp_leaf_to_spine_map = self.connection_manager_.update_leaf_to_spine_map_according_to_chosen_leaf_and_spine_for_large_job(chosen_leaf_id_num_list, choosed_spine_index_list, gpu_num,job_allocated_oxc_spine_link,job_used_spine_port_num_pair)[1]
                new_job.start_time = sim_time
                new_job.allocated_gpus = chosen_gpu_list
                new_job.job_leaf_to_spine_map = temp_leaf_to_spine_map
                new_job.allocated_oxc_spine_link = job_allocated_oxc_spine_link
                new_job.used_spine_port_num_pair = job_used_spine_port_num_pair
                self.current_job_list[job_id] = new_job
                allocation_link_mapping,record_leaf_num_map,record_spine_num_map = self.translate_updated_links(chosen_gpu_list, job_allocated_oxc_spine_link)
                print("finish allocation assign whole clos for small job")
                f2 = open('queue_length.txt','a')
                f2.write(str(len(queued_jobs)))
                f2.write(",")
                f2.write(str(sim_time) )
                f2.write("\n" )
                f2.close()                        
                assert len(chosen_gpu_list) == gpu_num
                self.check_spine()
                new_job.check_job_allocation_valid()
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
      
        print("start stage 3")
        choosed_spine_index_list = choose_group_in_spine_result[1]
        temp_spine_index = 0
        if(choose_group_in_spine_result[0]):
            for chosen_spine_id in choosed_spine_index_list:
                chosen_group_size = choose_group_in_spine_result[2][temp_spine_index]
                temp_spine_index += 1
                self.spine_resource_manager_.release_spine_group_with_give_id_and_group(chosen_spine_id, chosen_group_size)
        temp_z = pow(2,int(math.log2(int(gpu_num))))
        # 情况二：跨leaf通信但不需要spine迁移
        self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
        self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
        job_allocated_oxc_spine_link = {}
        job_used_spine_port_num_pair = {}


        require_leaf_num = int(gpu_num/self.gpu_per_leaf) #TODO
        temp_require_leaf_num = max(1,int(temp_z/self.gpu_per_leaf))
        allocate_success = False
        need_spine_migration = False
        try_time = 0
        #while((temp_require_leaf_num>=1 and not twopart(gpu_num)) or (temp_require_leaf_num<=self.leaf_num and twopart(gpu_num))):
        while(temp_require_leaf_num<=self.leaf_num and temp_require_leaf_num<=temp_z):
            try_time += 1
            temp_require_spine_num = int(temp_z/temp_require_leaf_num)
            leaf_remain_empt_gpu_list = []
            leaf_remain_empt_server_list = []
            for temp_leaf_id in range(self.leaf_num):
                leaf_remain_empt_server_list.append(0)
                leaf_remain_empt_gpu_list.append(0)
            for temp_server_id in range(self.server_num):
                temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
                if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
                    leaf_remain_empt_server_list[temp_leaf_id] += 1
                leaf_remain_empt_gpu_list[temp_leaf_id] += sum(self.server_resource_manager_.server_list[temp_server_id].gpu_group)
            spine_remain_empt_port_list = self.spine_resource_manager_.get_spine_remain_empt_port_list()  
            self.check_spine()
            allocate_success, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, temp_need_spine_migration, leaf_remain_gpu_num_map = self.connection_manager_.find_valid_gpu_for_no_pow2_task(gpu_num, leaf_remain_empt_server_list, spine_remain_empt_port_list, temp_require_leaf_num, temp_require_spine_num, job_id, leaf_remain_empt_gpu_list)
            need_spine_migration = need_spine_migration and temp_need_spine_migration
            if allocate_success:
                chosen_gpu_list = []
                remain_chosen_gpu_list = []
                leaf_port_num = 0
                for chosen_leaf_id in leaf_occupy_gpu_num_map:
                    temp_server_num = int(leaf_occupy_gpu_num_map[chosen_leaf_id]/self.gpu_per_server)
                    leaf_port_num += leaf_occupy_gpu_num_map[chosen_leaf_id]
                    for t in range(temp_server_num):
                        chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf(chosen_leaf_id, self.gpu_per_server))
                    self.leaf_resource_manager_.leaf_list[chosen_leaf_id].update_leaf_group_with_required_num(leaf_occupy_gpu_num_map[chosen_leaf_id])
                for chosen_leaf_id in leaf_remain_gpu_num_map:
                    leaf_port_num += leaf_remain_gpu_num_map[chosen_leaf_id]
                    remain_chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf_eleminating_fragmentation(chosen_leaf_id, leaf_remain_gpu_num_map[chosen_leaf_id]))
                    self.leaf_resource_manager_.leaf_list[chosen_leaf_id].update_leaf_group_with_required_num(leaf_remain_gpu_num_map[chosen_leaf_id])
                assert leaf_port_num == gpu_num
                spine_port_num = 0
                for chosen_spine_id in spine_occupy_port_num_map:
                    spine_port_num += spine_occupy_port_num_map[chosen_spine_id]
                    self.spine_resource_manager_.spine_list[chosen_spine_id].update_spine_group_with_required_num(spine_occupy_port_num_map[chosen_spine_id])
                assert spine_port_num == temp_z
                fus_gpu_list = []
                fus_gpu_list.extend(chosen_gpu_list)
                fus_gpu_list.extend(remain_chosen_gpu_list)
                new_job.start_time = sim_time
                new_job.allocated_gpus = fus_gpu_list
                new_job.job_leaf_to_spine_map = job_leaf_to_spine_map
                new_job.allocated_oxc_spine_link = job_oxc_leaf_spine_map
                new_job.used_spine_port_num_pair = spine_occupy_port_num_map
                self.current_job_list[job_id] = new_job
                allocation_link_mapping,record_leaf_num_map,record_spine_num_map = self.translate_updated_links(chosen_gpu_list, job_oxc_leaf_spine_map, remain_chosen_gpu_list)
                remain_comm_pair = self.fusion_gpu_list(chosen_gpu_list, remain_chosen_gpu_list)
                print("finish stage3 ",gpu_num)
                assert len(chosen_gpu_list) == temp_z
                f2 = open('queue_length.txt','a')
                f2.write(str(len(queued_jobs)))
                f2.write(",")
                f2.write(str(sim_time) )
                f2.write("\n" )
                f2.close()                                
                self.check_spine()
                time_end = time.perf_counter()
                time_sum = time_end-time_start
                
                Simulator.SCHEDULER_TIME_COST[job_id] = 0
                f3 = open('schedule_time_cost.txt','a')
                f3.write(str(job_id))
                f3.write(",")
                f3.write(str(time_sum) )
                f3.write("\n" )
                f3.close()  
                f3 = open('schedule_time_cost_m*n.txt','a')
                f3.write(str(job_id))
                f3.write(",")
                f3.write(str(time_sum) )
                f3.write("\n" )
                f3.close() 
                return True, True, fus_gpu_list, allocation_link_mapping,remain_comm_pair,None,None
            else:
                temp_require_leaf_num*=2
        if not allocate_success and need_spine_migration:
            print("finish allocation, no resource start migration ",gpu_num)
            f1 = open('fragmention.txt','a')
            f1.write(str(job_id) )
            f1.write(",")
            f1.write(str(sim_time) )
            f1.write("\n" )
            f1.close()
            self.spine_resource_manager_.print_remain_spoine_port_num()
            self.leaf_resource_manager_.print_remain_leaf_port_num()
            return False, None, None, None,None,None,None
        elif not allocate_success and not need_spine_migration:
            print("finish allocation, no resource due to locality2 ",gpu_num)
            f1 = open('fragmention.txt','a')
            f1.write(str(job_id) )
            f1.write(",")
            f1.write(str(sim_time) )
            f1.write("\n" )
            f1.close()
            self.spine_resource_manager_.print_remain_spoine_port_num()
            self.leaf_resource_manager_.print_remain_leaf_port_num()
            return False, None, None, None,None,None,None
        
        # # 情况三：跨leaf通信且需要spine迁移
        else:
            self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
            self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
            self.spine_resource_manager_.print_remain_spoine_port_num()
            self.leaf_resource_manager_.print_remain_leaf_port_num()
            print("no resource in stage 3",temp_z)
            return False, None, None, None,None,None,None
                
    def update_finished_job(self, job_id, sim_time, queued_jobs):
        print("some job finish" + str(job_id))
        if (1-self.server_resource_manager_.cal_remain_gpu_num()/2048)>=1:
            print("debug fuckfuckfuck")
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
        

    def translate_updated_links(self, gpu_indexes, updated_links, remain_chosen_gpu_list = []):
        record_leaf_num_map = {}
        record_spine_num_map = {}
        allocation_link_mapping = []
        # gpu - leaf links
        for output_gpu_index in gpu_indexes:
            output_leaf_index = utils.get_leaf_module_id(int(output_gpu_index/self.gpu_per_leaf), self.gpu_num)
            allocation_link_mapping.append((output_gpu_index, output_leaf_index, 1))
            allocation_link_mapping.append((output_leaf_index, output_gpu_index, 1))
        for output_gpu_index in remain_chosen_gpu_list:
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


            
            

# import copy
# import math
# import re
# import time
# from tabnanny import check
# import rapidAIsim.scheduler.locality_scheduler.utils as utils
# import rapidAIsim.scheduler.locality_scheduler.job as job
# import rapidAIsim.scheduler.locality_scheduler.connection_manager as connection_manager
# import rapidAIsim.scheduler.locality_scheduler.leaf_resource_manager as leaf_resource_manager
# import rapidAIsim.scheduler.locality_scheduler.server_resource_manager as server_resource_manager
# import rapidAIsim.scheduler.locality_scheduler.spine_resource_manager as spine_resource_manager


# def twopart(n):
#     return n&(n-1) == 0


# class GpuPlacementer:
#     def __init__(self,  spine_switch_num, leaf_switch_num, spine_switch_port_num, leaf_switch_port_num, server_num, oxc_num = 32):
#         self.gpu_num = spine_switch_num*spine_switch_port_num
#         self.server_num = server_num
#         self.leaf_num = leaf_switch_num
#         self.spine_num = spine_switch_num
#         self.oxc_num = oxc_num
#         self.gpu_per_server = int(self.gpu_num/server_num)
#         self.gpu_per_leaf = int(self.gpu_num/self.leaf_num)
#         self.port_per_spine = int(self.gpu_num/self.spine_num)
#         print("Cluster Info:")
#         print("server_num: "+" ")
#         print(server_num)
#         print("leaf_num: "+" ")
#         print(leaf_switch_num)
#         print("gpu_num: "+" ")
#         print(self.gpu_num)
#         self.server_resource_manager_ = server_resource_manager.ServerResourceManager(server_num, self.gpu_per_server, self.leaf_num)
#         self.leaf_resource_manager_ = leaf_resource_manager.LeafResourceManager(self.leaf_num, self.gpu_per_leaf)
#         self.spine_resource_manager_ = spine_resource_manager.SpineSwitchManager(self.spine_num, self.port_per_spine, [])
#         self.connection_manager_ = connection_manager.ConnectionManager(self.gpu_num, self.server_num, self.leaf_num, self.spine_num, self.oxc_num)

#         # job queue
#         self.current_job_list = {}
#         self.history_job_list = {}
        
#     def fusion_gpu_list(self, pow_2_gpu_list, remain_gpu_list):
#         communication_pair_list = []
#         fus_gpu_list = []
#         fus_gpu_list.extend(pow_2_gpu_list)
#         fus_gpu_list.extend(remain_gpu_list)
#         gpu_global_local_index_map = {}
#         for i in range(len(fus_gpu_list)):
#             gpu_global_local_index_map[fus_gpu_list[i]] = i
#         leaf_pow_2_gpu_list_map = {}
#         for pow_2_gpu in pow_2_gpu_list:
#             leaf_id = int(pow_2_gpu/self.gpu_per_leaf)
#             if leaf_id not in leaf_pow_2_gpu_list_map:
#                 leaf_pow_2_gpu_list_map[leaf_id] = []
#             leaf_pow_2_gpu_list_map[leaf_id].append(pow_2_gpu)
#         leaf_remain_gpu_list_map = {}
#         for remain_gpu in remain_gpu_list:
#             leaf_id = int(remain_gpu/self.gpu_per_leaf)
#             if leaf_id not in leaf_remain_gpu_list_map:
#                 leaf_remain_gpu_list_map[leaf_id] = []
#             leaf_remain_gpu_list_map[leaf_id].append(remain_gpu)
#         for leaf_id in leaf_remain_gpu_list_map:
#             for remain_gpu in leaf_remain_gpu_list_map[leaf_id]:
#                 to_comm_gpu = leaf_pow_2_gpu_list_map[leaf_id][0]
#                 communication_pair_list.append((gpu_global_local_index_map[remain_gpu], gpu_global_local_index_map[to_comm_gpu]))
#                 del(leaf_pow_2_gpu_list_map[leaf_id][0])
#         return communication_pair_list

#     def schedule(self, gpu_num, job_id, sim_time, queued_jobs, spine_strategy_mode ="gpu_first", oxc_need_reconfig = False):
#         from rapidAIsim.core.simulator import Simulator
#         print("some job arrive: "+str(job_id)+","+str(gpu_num), (1-self.server_resource_manager_.cal_remain_gpu_num()/2048))
#         time_start = time.perf_counter()
#         new_job = job.Job(job_id)
#         chosen_gpu_list = []
#         allocation_link_mapping = []
#         # 情况零：GPU数量不足
#         if gpu_num > self.server_resource_manager_.cal_remain_gpu_num():
#             print("finish allocation, no resource due to GPU")
#             return False, None, None, None,None,None,None
#         if not self.server_resource_manager_.whether_can_find_valid_server(gpu_num):
#             print("finish allocation, no resource due to locality")
#             return False, None, None, None,None,None,None
#         # 情况一：尝试不跨leaf通信 
#         print("Stage 1")
#         potentional_leaf_list = []
#         # Step1. 在leaf_resource_manager中选取合适的leafgroup
#         for temp_leaf_id in range(self.leaf_num):
#             require_server_num = math.ceil(gpu_num/self.gpu_per_server)
#             require_gpu_num_in_server = min(self.gpu_per_server,gpu_num)
#             valid_server_num = 0
#             for temp_server_num in range(int(temp_leaf_id*self.gpu_per_leaf/self.gpu_per_server), int((1+temp_leaf_id)*self.gpu_per_leaf/self.gpu_per_server),1):
#                 if self.server_resource_manager_.server_list[temp_server_num].remain_gpu_num()>=require_gpu_num_in_server:
#                     valid_server_num += 1
#             if valid_server_num>=require_server_num:
#                 potentional_leaf_list.append([temp_leaf_id, sum(self.leaf_resource_manager_.leaf_list[temp_leaf_id].leaf_group)])
#         potentional_leaf_list.sort( key=lambda x: (x[1])) 
#         if len(potentional_leaf_list)>0:
#             temp_leaf_id = potentional_leaf_list[0][0]
#             #  Step2 在选择的leaf交换机下联的server中按照locality选择gpu
#             chosen_gpu_list = self.server_resource_manager_.choose_gpu_in_one_leaf(temp_leaf_id, gpu_num)
#             self.leaf_resource_manager_.leaf_list[temp_leaf_id].update_leaf_group_with_required_num(gpu_num)
#             # gpu - leaf links
#             for output_gpu_index in chosen_gpu_list:
#                 assert int(output_gpu_index/self.gpu_per_leaf) == temp_leaf_id
#                 output_leaf_index = utils.get_leaf_module_id(temp_leaf_id, self.gpu_num)
#                 allocation_link_mapping.append([output_gpu_index, output_leaf_index, 1])
#                 allocation_link_mapping.append([output_leaf_index, output_gpu_index, 1])
#             #  记录job
#                 new_job.start_time = sim_time
#                 new_job.allocated_gpus = chosen_gpu_list
#                 self.current_job_list[job_id] = new_job
#             print("finish allocation one leaf")
#             self.check_spine()
#             f2 = open('queue_length.txt','a')
#             f2.write(str(len(queued_jobs)))
#             f2.write(",")
#             f2.write(str(sim_time) )
#             f2.write("\n" )
#             f2.close()
#             time_end = time.perf_counter()
#             time_sum = time_end-time_start
            
#             Simulator.SCHEDULER_TIME_COST[job_id] = 0
#             f3 = open('schedule_time_cost.txt','a')
#             f3.write(str(job_id))
#             f3.write(",")
#             f3.write(str(time_sum) )
#             f3.write("\n" )
#             f3.close()  
#             return True, True, chosen_gpu_list, allocation_link_mapping,None,None,None
#         # 否则需要跨leaf通信
#         choose_group_in_spine_result = self.spine_resource_manager_.choose_group_in_spine(gpu_num)
#         if(choose_group_in_spine_result[0] and len(choose_group_in_spine_result[1])==1) and  twopart(gpu_num):
#             # 尽可能在同一个spine
#             print("stage 2")
#             choosed_spine_index_list = choose_group_in_spine_result[1]
#             self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
#             self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
#             leaf_remain_empt_server_list = []
#             for temp_leaf_id in range(self.leaf_num):
#                 leaf_remain_empt_server_list.append(0)
#             for temp_server_id in range(self.server_num):
#                 temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
#                 if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
#                     leaf_remain_empt_server_list[temp_leaf_id] += 1
#             chosen_gpu_list = []
#             job_allocated_oxc_spine_link = {}
#             job_used_spine_port_num_pair = {}
#             temp_spine_index = 0
#             for chosen_spine_id in choosed_spine_index_list:
#                 chosen_group_size = choose_group_in_spine_result[2][temp_spine_index]
#                 temp_spine_index += 1
#                 if chosen_group_size != int(gpu_num/len(choosed_spine_index_list)):
#                     print("fuck", choose_group_in_spine_result[2], gpu_num)
#                 valid, server_occupy_gpuNum_map = self.connection_manager_.find_valid_gpu_for_specific_spine(chosen_group_size, chosen_spine_id, self.server_resource_manager_.return_server_remain_gpuNum_map(),job_allocated_oxc_spine_link,job_used_spine_port_num_pair, leaf_remain_empt_server_list)
#                 if(not valid):
#                     self.spine_resource_manager_.release_spine_group_with_give_id_and_group(chosen_spine_id, chosen_group_size)
#                     print("finish allocation, no resource due to locality3", len(choosed_spine_index_list), gpu_num)
#                     self.leaf_resource_manager_.print_remain_leaf_port_num()
#                     self.spine_resource_manager_.print_remain_spoine_port_num()
#                     self.spine_resource_manager_.print_resource_info()
#                     return False, None, None, None,None,None,None
#                 for server_id in server_occupy_gpuNum_map:
#                     if server_occupy_gpuNum_map[server_id]>0:
#                         chosen_gpu_list.extend(self.server_resource_manager_.server_list[server_id].occupy_gpu_with_required_num(server_occupy_gpuNum_map[server_id])[1])
#             chosen_leaf_id_num_list = self.leaf_resource_manager_.update_group_with_given_gpu_list(chosen_gpu_list)

#             temp_leaf_to_spine_map = {} # key 为leaf的index，value为另一个map B， map B的key为spine交换机的index，value为该leaf要新连多少根线到该spine
#             for choosed_leaf_id_num_pair in chosen_leaf_id_num_list:
#                 temp_leaf_to_each_spine_map = {}
#                 for choosed_spine_index in choosed_spine_index_list:
#                     temp_leaf_to_each_spine_map[choosed_spine_index] = int(choosed_leaf_id_num_pair[1]/len(choosed_spine_index_list))
#                 temp_leaf_to_spine_map[choosed_leaf_id_num_pair[0]] = temp_leaf_to_each_spine_map

#             new_job.start_time = sim_time
#             new_job.allocated_gpus = chosen_gpu_list
#             new_job.job_leaf_to_spine_map = temp_leaf_to_spine_map
#             new_job.allocated_oxc_spine_link = job_allocated_oxc_spine_link
#             new_job.used_spine_port_num_pair = job_used_spine_port_num_pair
#             self.current_job_list[job_id] = new_job
#             allocation_link_mapping,record_leaf_num_map,record_spine_num_map = self.translate_updated_links(chosen_gpu_list, job_allocated_oxc_spine_link)
#             print("finish allocation assign whole clos for large job")
#             self.check_spine()
#             f2 = open('queue_length.txt','a')
#             f2.write(str(len(queued_jobs)))
#             f2.write(",")
#             f2.write(str(sim_time) )
#             f2.write("\n" )
#             f2.close()     
#             if len(chosen_gpu_list) != gpu_num:
#                 print(len(chosen_gpu_list), gpu_num, len(choosed_spine_index_list))
#             assert len(chosen_gpu_list) == gpu_num
#             time_end = time.perf_counter()
#             time_sum = time_end-time_start
            
#             Simulator.SCHEDULER_TIME_COST[job_id] = 0
#             f3 = open('schedule_time_cost.txt','a')
#             f3.write(str(job_id))
#             f3.write(",")
#             f3.write(str(time_sum) )
#             f3.write("\n" )
#             f3.close()  
#             return True, True, chosen_gpu_list, allocation_link_mapping,None,None,None
#         elif choose_group_in_spine_result[0] and len(choose_group_in_spine_result[1])>1 and twopart(gpu_num):
#             print("start whole leaf-spine")
#             self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
#             self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
#             job_allocated_oxc_spine_link = {}
#             job_used_spine_port_num_pair = {}

#             choosed_spine_index_list = choose_group_in_spine_result[1]
#             require_leaf_num = int(gpu_num/self.gpu_per_leaf) #TODO
#             chosen_leaf_id_num_list = []
#             for leaf_spine in self.leaf_resource_manager_.leaf_list:
#                 if self.gpu_per_leaf in leaf_spine.leaf_group and len(chosen_leaf_id_num_list)<require_leaf_num:
#                     chosen_leaf_id_num_list.append([leaf_spine.leaf_id,self.gpu_per_leaf])
#             if len(chosen_leaf_id_num_list)>=require_leaf_num:
#                 print(print("stage 2.2"))
#                 chosen_gpu_list = []
#                 for chosen_leaf_id_num in chosen_leaf_id_num_list:
#                     chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf(chosen_leaf_id_num[0], self.gpu_per_leaf))
#                     self.leaf_resource_manager_.leaf_list[chosen_leaf_id_num[0]].update_leaf_group_with_required_num(self.gpu_per_leaf)
#                 print("debug chosen_leaf_id_num_list", chosen_leaf_id_num_list)
#                 print("debug choosed_spine_index_list", choosed_spine_index_list)
#                 print("debug job_allocated_oxc_spine_link", job_allocated_oxc_spine_link)
#                 print("debug job_used_spine_port_num_pair", job_used_spine_port_num_pair)
#                 temp_leaf_to_spine_map = self.connection_manager_.update_leaf_to_spine_map_according_to_chosen_leaf_and_spine_for_large_job(chosen_leaf_id_num_list, choosed_spine_index_list, gpu_num,job_allocated_oxc_spine_link,job_used_spine_port_num_pair)[1]
#                 new_job.start_time = sim_time
#                 new_job.allocated_gpus = chosen_gpu_list
#                 new_job.job_leaf_to_spine_map = temp_leaf_to_spine_map
#                 new_job.allocated_oxc_spine_link = job_allocated_oxc_spine_link
#                 new_job.used_spine_port_num_pair = job_used_spine_port_num_pair
#                 self.current_job_list[job_id] = new_job
#                 allocation_link_mapping,record_leaf_num_map,record_spine_num_map = self.translate_updated_links(chosen_gpu_list, job_allocated_oxc_spine_link)
#                 print("finish allocation assign whole clos for small job")
#                 f2 = open('queue_length.txt','a')
#                 f2.write(str(len(queued_jobs)))
#                 f2.write(",")
#                 f2.write(str(sim_time) )
#                 f2.write("\n" )
#                 f2.close()                        
#                 assert len(chosen_gpu_list) == gpu_num
#                 self.check_spine()
#                 new_job.check_job_allocation_valid()
#                 time_end = time.perf_counter()
#                 time_sum = time_end-time_start
                
#                 Simulator.SCHEDULER_TIME_COST[job_id] = 0
#                 f3 = open('schedule_time_cost.txt','a')
#                 f3.write(str(job_id))
#                 f3.write(",")
#                 f3.write(str(time_sum) )
#                 f3.write("\n" )
#                 f3.close()  
#                 return True, True, chosen_gpu_list, allocation_link_mapping,None,None,None
      
#         print("start stage 3")
#         choosed_spine_index_list = choose_group_in_spine_result[1]
#         temp_spine_index = 0
#         if(choose_group_in_spine_result[0]):
#             for chosen_spine_id in choosed_spine_index_list:
#                 chosen_group_size = choose_group_in_spine_result[2][temp_spine_index]
#                 temp_spine_index += 1
#                 self.spine_resource_manager_.release_spine_group_with_give_id_and_group(chosen_spine_id, chosen_group_size)
#         temp_z = pow(2,int(math.log2(int(gpu_num))))
#         # 情况二：跨leaf通信但不需要spine迁移
#         self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
#         self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
#         job_allocated_oxc_spine_link = {}
#         job_used_spine_port_num_pair = {}


#         require_leaf_num = int(gpu_num/self.gpu_per_leaf) #TODO
#         temp_require_leaf_num = max(1,int(temp_z/self.gpu_per_leaf))
#         allocate_success = False
#         need_spine_migration = False
#         try_time = 0
#         #while((temp_require_leaf_num>=1 and not twopart(gpu_num)) or (temp_require_leaf_num<=self.leaf_num and twopart(gpu_num))):
#         while(temp_require_leaf_num<=self.leaf_num and temp_require_leaf_num<=temp_z):
#             try_time += 1
#             temp_require_spine_num = int(temp_z/temp_require_leaf_num)
#             leaf_remain_empt_gpu_list = []
#             leaf_remain_empt_server_list = []
#             for temp_leaf_id in range(self.leaf_num):
#                 leaf_remain_empt_server_list.append(0)
#                 leaf_remain_empt_gpu_list.append(0)
#             for temp_server_id in range(self.server_num):
#                 temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
#                 if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
#                     leaf_remain_empt_server_list[temp_leaf_id] += 1
#                 leaf_remain_empt_gpu_list[temp_leaf_id] += sum(self.server_resource_manager_.server_list[temp_server_id].gpu_group)
#             spine_remain_empt_port_list = self.spine_resource_manager_.get_spine_remain_empt_port_list()  
#             self.check_spine()
#             allocate_success, job_leaf_to_spine_map, job_oxc_leaf_spine_map, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, temp_need_spine_migration, leaf_remain_gpu_num_map = self.connection_manager_.find_valid_gpu_for_no_pow2_task(gpu_num, leaf_remain_empt_server_list, spine_remain_empt_port_list, temp_require_leaf_num, temp_require_spine_num, job_id, leaf_remain_empt_gpu_list)
#             need_spine_migration = need_spine_migration and temp_need_spine_migration
#             if allocate_success:
#                 chosen_gpu_list = []
#                 remain_chosen_gpu_list = []
#                 leaf_port_num = 0
#                 for chosen_leaf_id in leaf_occupy_gpu_num_map:
#                     temp_server_num = int(leaf_occupy_gpu_num_map[chosen_leaf_id]/self.gpu_per_server)
#                     leaf_port_num += leaf_occupy_gpu_num_map[chosen_leaf_id]
#                     for t in range(temp_server_num):
#                         chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf(chosen_leaf_id, self.gpu_per_server))
#                     self.leaf_resource_manager_.leaf_list[chosen_leaf_id].update_leaf_group_with_required_num(leaf_occupy_gpu_num_map[chosen_leaf_id])
#                 for chosen_leaf_id in leaf_remain_gpu_num_map:
#                     leaf_port_num += leaf_remain_gpu_num_map[chosen_leaf_id]
#                     remain_chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf_eleminating_fragmentation(chosen_leaf_id, leaf_remain_gpu_num_map[chosen_leaf_id]))
#                     self.leaf_resource_manager_.leaf_list[chosen_leaf_id].update_leaf_group_with_required_num(leaf_remain_gpu_num_map[chosen_leaf_id])
#                 assert leaf_port_num == gpu_num
#                 spine_port_num = 0
#                 for chosen_spine_id in spine_occupy_port_num_map:
#                     spine_port_num += spine_occupy_port_num_map[chosen_spine_id]
#                     self.spine_resource_manager_.spine_list[chosen_spine_id].update_spine_group_with_required_num(spine_occupy_port_num_map[chosen_spine_id])
#                 assert spine_port_num == temp_z
#                 fus_gpu_list = []
#                 fus_gpu_list.extend(chosen_gpu_list)
#                 fus_gpu_list.extend(remain_chosen_gpu_list)
#                 new_job.start_time = sim_time
#                 new_job.allocated_gpus = fus_gpu_list
#                 new_job.job_leaf_to_spine_map = job_leaf_to_spine_map
#                 new_job.allocated_oxc_spine_link = job_oxc_leaf_spine_map
#                 new_job.used_spine_port_num_pair = spine_occupy_port_num_map
#                 self.current_job_list[job_id] = new_job
#                 allocation_link_mapping,record_leaf_num_map,record_spine_num_map = self.translate_updated_links(chosen_gpu_list, job_oxc_leaf_spine_map, remain_chosen_gpu_list)
#                 remain_comm_pair = self.fusion_gpu_list(chosen_gpu_list, remain_chosen_gpu_list)
#                 print("finish stage3 ",gpu_num)
#                 assert len(chosen_gpu_list) == temp_z
#                 f2 = open('queue_length.txt','a')
#                 f2.write(str(len(queued_jobs)))
#                 f2.write(",")
#                 f2.write(str(sim_time) )
#                 f2.write("\n" )
#                 f2.close()                                
#                 self.check_spine()
#                 time_end = time.perf_counter()
#                 time_sum = time_end-time_start
                
#                 Simulator.SCHEDULER_TIME_COST[job_id] = 0
#                 f3 = open('schedule_time_cost.txt','a')
#                 f3.write(str(job_id))
#                 f3.write(",")
#                 f3.write(str(time_sum) )
#                 f3.write("\n" )
#                 f3.close()  
#                 f3 = open('schedule_time_cost_m*n.txt','a')
#                 f3.write(str(job_id))
#                 f3.write(",")
#                 f3.write(str(time_sum) )
#                 f3.write("\n" )
#                 f3.close() 
#                 return True, True, fus_gpu_list, allocation_link_mapping,remain_comm_pair,None,None
#             else:
#                 temp_require_leaf_num*=2
#         if not allocate_success and need_spine_migration:
#             print("finish allocation, no resource start migration ",gpu_num)
#             f1 = open('fragmention.txt','a')
#             f1.write(str(job_id) )
#             f1.write(",")
#             f1.write(str(sim_time) )
#             f1.write("\n" )
#             f1.close()
#             self.spine_resource_manager_.print_remain_spoine_port_num()
#             self.leaf_resource_manager_.print_remain_leaf_port_num()
#             return False, None, None, None,None,None,None
#         elif not allocate_success and not need_spine_migration:
#             print("finish allocation, no resource due to locality2 ",gpu_num)
#             f1 = open('fragmention.txt','a')
#             f1.write(str(job_id) )
#             f1.write(",")
#             f1.write(str(sim_time) )
#             f1.write("\n" )
#             f1.close()
#             self.spine_resource_manager_.print_remain_spoine_port_num()
#             self.leaf_resource_manager_.print_remain_leaf_port_num()
#             return False, None, None, None,None,None,None
        
#         # # 情况三：跨leaf通信且需要spine迁移
#         else:
#             self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
#             self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
#             self.spine_resource_manager_.print_remain_spoine_port_num()
#             self.leaf_resource_manager_.print_remain_leaf_port_num()
#             print("no resource in stage 3",temp_z)
#             return False, None, None, None,None,None,None
                
#     def update_finished_job(self, job_id, sim_time, queued_jobs):
#         print("some job finish" + str(job_id), (1-self.server_resource_manager_.cal_remain_gpu_num()/2048))
#         if (1-self.server_resource_manager_.cal_remain_gpu_num()/2048)>=1:
#             print("debug fuckfuckfuck")
#         to_leave_job = copy.deepcopy(self.current_job_list[job_id])
#         to_leave_job.finish_time = sim_time
#         self.history_job_list[job_id] = to_leave_job
#         self.server_resource_manager_.release_gpu_in_server(to_leave_job.allocated_gpus)
#         self.leaf_resource_manager_.release_group_with_given_gpu_list(to_leave_job.allocated_gpus)
#         spine_portNum_map = {}
#         for oxc_id in to_leave_job.allocated_oxc_spine_link:
#             for leaf_id in to_leave_job.allocated_oxc_spine_link[oxc_id]:
#                 spine_id = to_leave_job.allocated_oxc_spine_link[oxc_id][leaf_id]
#                 if spine_id not in spine_portNum_map:
#                     spine_portNum_map[spine_id] = 0
#                 spine_portNum_map[spine_id] += 1
#         for spine_id in spine_portNum_map:
#             self.spine_resource_manager_.release_spine_group_with_give_id_and_group(spine_id, spine_portNum_map[spine_id])
#         self.connection_manager_.release_connection_resource(to_leave_job.allocated_oxc_spine_link)
#         del self.current_job_list[job_id]
#         f2 = open('queue_length.txt','a')
#         f2.write(str(len(queued_jobs)))
#         f2.write(",")
#         f2.write(str(sim_time) )
#         f2.write("\n" )
#         f2.close()        
#         self.check_spine()
    
#     def check_spine(self):
#         temp_size = 0
#         for temp_job_key in self.current_job_list:
#             temp_job = self.current_job_list[temp_job_key]
#             for chosen_spine_id in temp_job.used_spine_port_num_pair:
#                 temp_size+=temp_job.used_spine_port_num_pair[chosen_spine_id]
#         # print(self.gpu_num-temp_size, self.spine_resource_manager_.cal_remain_spoine_port_num())
#         assert(self.gpu_num-temp_size==self.spine_resource_manager_.cal_remain_spoine_port_num())
        

#     def translate_updated_links(self, gpu_indexes, updated_links, remain_chosen_gpu_list = []):
#         record_leaf_num_map = {}
#         record_spine_num_map = {}
#         allocation_link_mapping = []
#         # gpu - leaf links
#         for output_gpu_index in gpu_indexes:
#             output_leaf_index = utils.get_leaf_module_id(int(output_gpu_index/self.gpu_per_leaf), self.gpu_num)
#             allocation_link_mapping.append((output_gpu_index, output_leaf_index, 1))
#             allocation_link_mapping.append((output_leaf_index, output_gpu_index, 1))
#         for output_gpu_index in remain_chosen_gpu_list:
#             output_leaf_index = utils.get_leaf_module_id(int(output_gpu_index/self.gpu_per_leaf), self.gpu_num)
#             allocation_link_mapping.append((output_gpu_index, output_leaf_index, 1))
#             allocation_link_mapping.append((output_leaf_index, output_gpu_index, 1))
#         # leaf - spine links
#         temp_leaf_to_spine_num = {}
#         for oxc_id in updated_links:
#             for leaf_id in updated_links[oxc_id]:
#                 spine_id = updated_links[oxc_id][leaf_id]
#                 if (leaf_id,spine_id) not in temp_leaf_to_spine_num:
#                     temp_leaf_to_spine_num[(leaf_id,spine_id)] = 0
#                 temp_leaf_to_spine_num[(leaf_id,spine_id)] += 1

#                 if leaf_id not in record_leaf_num_map:
#                     record_leaf_num_map[leaf_id] = 0
#                 record_leaf_num_map[leaf_id] += 1
#                 if spine_id not in record_spine_num_map:
#                     record_spine_num_map[spine_id] = 0
#                 record_spine_num_map[spine_id] += 1
#         for leaf_spine_pair in temp_leaf_to_spine_num:
#             allocation_link_mapping.append((leaf_spine_pair[0]+self.gpu_num, leaf_spine_pair[1]+self.gpu_num+self.leaf_num, temp_leaf_to_spine_num[leaf_spine_pair]))
#             allocation_link_mapping.append((leaf_spine_pair[1]+self.gpu_num+self.leaf_num,leaf_spine_pair[0]+self.gpu_num, temp_leaf_to_spine_num[leaf_spine_pair]))
#         return allocation_link_mapping,record_leaf_num_map,record_spine_num_map


            
            