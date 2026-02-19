import rapidAIsim.scheduler.hw_new_arg_sim_compare.server
import rapidAIsim.scheduler.hw_new_arg_sim_compare.job as job
import rapidAIsim.scheduler.hw_new_arg_sim_compare.ele_group as ele_group
import gurobipy


class NaiveSchedulerCompare:
    def __init__(self, tor_num=512, ele_group_num=4):
        # self.gpu_num = gpu_num # 8192
        # self.oxc_num = oxc_num #4
        # self.ele_num_per_group = ele_num #1
        self.total_tor_num = tor_num
        self.ele_group_num = ele_group_num
        self.ele_list = [ele_group.ELE_Group(i,int(tor_num/self.ele_group_num)) for i in range(self.ele_group_num)]
        self.job_id_map= {}
    
    def update_finished_job(self, job_id, sim_time, queued_jobs):
        free_gpu = 0
        for ele in self.ele_list:
            free_gpu += ele.remain_tor_in_this_group
        f2 = open('gpu_utilization.txt','a')
        f2.write(str(1-free_gpu/4096))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()
        cur_job = self.job_id_map[job_id]
        self.job_id_map[job_id].finish_time = sim_time
        for global_tor_id in cur_job.allocated_tors:
            ele_id = int(global_tor_id/int(self.total_tor_num/self.ele_group_num))
            local_tor_id = global_tor_id%int(self.total_tor_num/self.ele_group_num)
            self.ele_list[ele_id].release_tor(local_tor_id)
            
    def schedule(self, used_tor_num, job_id, sim_time, queued_jobs, strict_clos, check_conflict):
        free_gpu = 0
        for ele in self.ele_list:
            free_gpu += ele.remain_tor_in_this_group
        f2 = open('gpu_utilization.txt','a')
        f2.write(str(1-free_gpu/4096))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()
        tor_occupy_list = []
        allocated_link_mapping = []
        cur_job = job.Job(job_id, used_tor_num)
        allocate_success = False
        potention_valid_ele_list = []
        for tmp_ele in self.ele_list:
            if tmp_ele.remain_tor_in_this_group >= used_tor_num:
                potention_valid_ele_list.append(tmp_ele)
        if len(potention_valid_ele_list)>0:
            #potention_valid_ele_list.sort(key=lambda x: x.remain_tor_in_this_group, reverse=True)
            #potention_valid_ele_list.sort(key=lambda x: x.remain_tor_in_this_group, reverse=False)
            tor_occupy_list = potention_valid_ele_list[0].occupy_tor(used_tor_num)
            allocate_success = True
            cur_job.allocated_tors = tor_occupy_list
            
        if allocate_success:
            cur_job.start_time = sim_time
            self.job_id_map[job_id] = cur_job
        else:
            if free_gpu>used_tor_num:
                print("no resource ",used_tor_num, free_gpu,len(potention_valid_ele_list))
        f2 = open('queue_length.txt','a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()
        return allocate_success, tor_occupy_list, allocated_link_mapping, None, None
    
    