import copy
import gurobipy
import math
from collections import Counter

# 控制leaf资源的调度，包括group
# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，不涉及spine交换机
# 2. 当需要跨leaf通信时，将所有涉及的gpu连通到同一个spine，此时有两种情况
#    a.当存在某个spine拥有合适的group，那么在leaf和spine按full mesh连接，
#      返回更新的leaf_to_spine_map并进行整数规划
#    b.进行gpu迁移，这里只根据group判断某个spine迁移多少个端口到另一个spine
#      上，具体迁移哪个spine需要结合job信息

def _modify_to_2exponent(task_occupied_NIC_num):
    import math
    exponent = math.log2(task_occupied_NIC_num)
    exponent = math.ceil(exponent)
    exponent = max(1, exponent)
    task_occupied_NIC_num =  2 ** exponent

    return task_occupied_NIC_num

class SpineSwitch:
    def __init__(self, id, port_per_spine):
        self.port_per_spine = port_per_spine
        self.spine_id = id
        self.spine_group = [port_per_spine] #注意group与GPU的更新是否同步

    # 给定需要的端口数量，返回离他最近的group的大小,返回的group大于等于required_group_size，
    # 如果没有找到返回-1
    def find_closest_spine_group_size(self, required_group_size):  
        cloest_group_size = 10000
        for exist_group_size in self.spine_group:
            # exist_group_size在至少要满足需求的同时要找最小的可行group
            if(exist_group_size >= required_group_size 
               and cloest_group_size > exist_group_size):
                cloest_group_size = exist_group_size
        if cloest_group_size == 10000:
            return [self.spine_group, -1]
        return [self.spine_id, cloest_group_size]

    # 根据需要的gpu大小，更新group
    def update_spine_group_with_required_num(self, require_port_num):
        have_chosen_gpu_num = int (math.pow( 2, int( math.log2(require_port_num) ) ))
        need_group_list = [have_chosen_gpu_num]
        temp_potentional_group_size = have_chosen_gpu_num
        while(have_chosen_gpu_num<require_port_num):
            if(have_chosen_gpu_num+int(temp_potentional_group_size) <= require_port_num):
                need_group_list.append(temp_potentional_group_size)
                have_chosen_gpu_num += temp_potentional_group_size
                temp_potentional_group_size = int(temp_potentional_group_size/2)
            else:
                temp_potentional_group_size = int(temp_potentional_group_size/2)
        for need_group in need_group_list:
            group_to_remove = need_group
            if need_group in self.spine_group:
                self.spine_group.remove(need_group)
            else:
                group_to_remove = need_group*2
                group_to_add = [need_group]
                while(group_to_remove not in self.spine_group):
                    group_to_add.append(group_to_remove)
                    group_to_remove*=2 #TODO maybe some wrong
                self.spine_group.remove(group_to_remove)
                self.spine_group.extend(group_to_add)
                assert group_to_remove<=self.port_per_spine
        return self.spine_id
    
    def clear_spine(self):
        self.spine_group = [self.port_per_spine]

    # 根据group大小释放资源，更新spine group
    def release_spine_group_with_required_num(self, require_gpu_num):
        have_released_gpu_num = int (math.pow( 2, int( math.log2(require_gpu_num) ) ))
        release_group_list = [have_released_gpu_num]
        temp_potentional_group_size = have_released_gpu_num
        while(have_released_gpu_num<require_gpu_num):
            if(have_released_gpu_num+int(temp_potentional_group_size) <= require_gpu_num):
                release_group_list.append(temp_potentional_group_size)
                have_released_gpu_num += temp_potentional_group_size
                temp_potentional_group_size = int(temp_potentional_group_size/2)
            else:
                temp_potentional_group_size = int(temp_potentional_group_size/2)
        for release_group in release_group_list:
            if release_group not in self.spine_group:
                self.spine_group.append(release_group)
            else:
                to_del_list = []
                multi_factor = 1
                while(multi_factor*release_group in self.spine_group):
                    to_del_list.append(multi_factor*release_group)
                    multi_factor*=2
                self.spine_group.append(multi_factor*release_group)
                for to_del_group in to_del_list:
                    self.spine_group.remove(to_del_group)
        assert sum(self.spine_group)<=self.port_per_spine

    # debug用的函数
    def print_resource_info(self):
        print("spine id: ",end=" ")
        print(self.spine_id)
        print("group state: ",end=" ")
        print(self.spine_group)

class SpineSwitchManager:
    def __init__(self, spine_num = 8, port_per_spine = 32, banned_spine_list= []):
        self.spine_num = spine_num
        self.port_per_spine = port_per_spine
        # 生成spine列表
        self.spine_list = [] # 禁止排序
        for spine_id in range(spine_num):
            temp_spine = SpineSwitch(spine_id, port_per_spine)
            self.spine_list.append(temp_spine)
        self.banned_spine_list = banned_spine_list
        
    def clear_spine_list(self):
        for spine in self.spine_list:
            spine.clear_spine()

    def cal_remain_spoine_port_num(self):
        remain_port_n = 0
        for spine in self.spine_list:
            remain_port_n += sum(spine.spine_group)
        return remain_port_n

    def print_remain_spoine_port_num(self):
        print("{",end="")
        for spine in self.spine_list:
            if spine.spine_group != []:
                print(spine.spine_id,end=": ")
                print(sum(spine.spine_group),end=", ")
        print()
        
    def print_resource_info(self):
        for spineI in self.spine_list:
            spineI.print_resource_info()

    def get_spine_remain_empt_port_list(self):
        spine_remain_empt_port_list = []
        for spine in self.spine_list:
            spine_remain_empt_port_list.append(sum(spine.spine_group))
        return spine_remain_empt_port_list


    # 根据需要的端口数量判断是否存在合适的spine group，如果是则返回true，spine id并更新group（具体端口选择
    # 在connection manager中进行），如果不存在则返回false以及spine迁移方案
    def choose_group_in_spine(self, require_gpu_num):
        need_group_size = min(self.port_per_spine, require_gpu_num)
        need_spine_num = math.ceil(require_gpu_num/need_group_size)
        potential_spineId_group_pair_list = []
        for temp_spine in self.spine_list:
            if temp_spine.spine_id not in self.banned_spine_list:
                group_in_this_spine = temp_spine.find_closest_spine_group_size(need_group_size)
                if group_in_this_spine[1]!=-1:
                    potential_spineId_group_pair_list.append(group_in_this_spine)
        # 如果有合适的spine交换机,那么根据spine交换机信息按选择的group大小从小到大排序，
        # 选择合适的spine并更新group
        if(len(potential_spineId_group_pair_list)>=need_spine_num):
            potential_spineId_group_pair_list.sort( key=lambda x: (x[1]) ) # 选择最小的符合条件的group
            choosed_spine_index_list = []
            choosed_spine_portnum_list = []
            chosen_spine_port_num = 0
            for have_chosen_spine_num in range(need_spine_num):
                choosed_spine = self.spine_list[potential_spineId_group_pair_list[have_chosen_spine_num][0]]
                choosed_spine.update_spine_group_with_required_num(min(need_group_size, (require_gpu_num-chosen_spine_port_num)))
                choosed_spine_portnum_list.append(min(need_group_size, (require_gpu_num-chosen_spine_port_num)))
                chosen_spine_port_num += min(need_group_size, (require_gpu_num-chosen_spine_port_num))
                choosed_spine_index_list.append(choosed_spine.spine_id)
            assert chosen_spine_port_num == require_gpu_num
            return True, choosed_spine_index_list, choosed_spine_portnum_list
        else:
            print("debug spine",need_group_size,need_spine_num)
            return False, None, None
    


    def release_spine_group_with_give_id_and_group(self, spine_id, group_size):
        self.spine_list[spine_id].release_spine_group_with_required_num(group_size)


    # def new_make_spine_migration_strategy(self):
    #     max_weight_spine = int(math.log2(self.port_per_spine))+1
    #     print("start migration")
    #     job_running_group_require = [0 for i in range(max_weight_spine)]
    #     job_group_require = [0 for i in range(max_weight_spine)]
    #     init_spine_group_list = [] 
    #     for spine in self.spine_list:
    #         temp_spine_group_vec = [0 for i in range(max_weight_spine)]
    #         for temp_group_size in spine.spine_group:
    #             assert temp_spine_group_vec[int( math.log2(temp_group_size) )] == 0
    #             temp_spine_group_vec[int( math.log2(temp_group_size) )] = 1
    #         init_spine_group_list.append(temp_spine_group_vec)

    #     spine_state_vec = []
    #     for temp_spine_state_vec in init_spine_group_list:
    #         spine_state_vec.extend(temp_spine_state_vec)
    #     running_group_list = []
    #     for temp_spine_group_id in range(len(init_spine_group_list)):
    #         temp_spine_group = init_spine_group_list[temp_spine_group_id]
    #         temp_running_group = [0 for i in range(max_weight_spine)]
    #         remain_gpu = self.port_per_spine
    #         for temp_element in range(len(temp_spine_group)):
    #             if(temp_spine_group[temp_element]):
    #                 remain_gpu -= int(pow(2,temp_element))
    #         if remain_gpu>0:
    #             have_chosen_group_size = int (math.pow( 2, int( math.log2(remain_gpu) ) ))
    #             job_running_group_require[int( math.log2(int (math.pow( 2, int( math.log2(remain_gpu) ) ))))]+=1
    #             job_group_require[int( math.log2(int (math.pow( 2, int( math.log2(remain_gpu) ) ))))]+=1
    #             temp_running_group[int(math.log2(have_chosen_group_size))] = 1
    #             temp_potentional_group_size = have_chosen_group_size
    #             while(have_chosen_group_size<remain_gpu):
    #                 if(have_chosen_group_size+int(temp_potentional_group_size) <= remain_gpu):
    #                     temp_running_group[int( math.log2(temp_potentional_group_size))] = 1
    #                     job_running_group_require[int( math.log2(int (math.pow( 2, int( math.log2(remain_gpu) ) ))))]+=1
    #                     job_group_require[int( math.log2(int (math.pow( 2, int( math.log2(remain_gpu) ) ))))]+=1
    #                     have_chosen_group_size += temp_potentional_group_size
    #                     temp_potentional_group_size = int(temp_potentional_group_size/2)
    #                 else:
    #                     temp_potentional_group_size = int(temp_potentional_group_size/2)
    #         else:
    #             assert remain_gpu == 0
    #         # print(job_running_group_require)
    #         # print(job_group_require)
    #         # print(self.spine_list[temp_spine_group_id].spine_group)
    #         # print(temp_spine_group)
    #         # print(temp_running_group)
    #         running_group_list.extend(temp_running_group)
    #     print("fuck")
    #     print(init_spine_group_list)
    #     print(job_running_group_require)
    #     print(job_group_require)
            
    #     m = gurobipy.Model("Clos solution")
    #     m.setParam('OutputFlag', 0)
    #     m.setParam('TimeLimit', 120)

    #     x_i = {}
    #     for it in range(len(spine_state_vec)):
    #         x_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i')
    #     r = {}
    #     for it in range(len(spine_state_vec)):
    #         r[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.port_per_spine,name='r_i')

    #     obj_val = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=4096,name='obj')
    #     obj_x_i = {}
    #     for it in range(len(spine_state_vec)):
    #         obj_x_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.port_per_spine,name='obj_x_i')
    #     obj_r_i = {}
    #     for it in range(len(spine_state_vec)):
    #         obj_r_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='obj_r_i')
    #     m.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
    #     m.update()

    #     # 设置条件
    #     for it in range(len(spine_state_vec)):
    #         m.addConstr(gurobipy.quicksum( (x_i[j] + r[j])*math.pow(2,int(j%max_weight_spine)) for j in range(len(spine_state_vec)) if int(it/max_weight_spine) == int(j/max_weight_spine)) == self.port_per_spine)

    #     for k in range(max_weight_spine):
    #         temp_class_list = []
    #         for temp_k in range(k,max_weight_spine,1):
    #             temp_class_list.append(temp_k)
    #         temp_value = 0
    #         for temp_k in range(k,max_weight_spine,1):
    #             temp_value += int(job_group_require[k]*pow(2,temp_k))
    #         m.addConstr(gurobipy.quicksum( (x_i[j] + r[j])*math.pow(2,int(j%max_weight_spine)) for j in range(len(spine_state_vec)) if int(j%max_weight_spine) in temp_class_list) >= temp_value)

    #     for k in range(max_weight_spine-1):
    #         m.addConstr(gurobipy.quicksum( (x_i[j] + r[j]) for j in range(len(spine_state_vec)) if int(k) == int(j%max_weight_spine)) <= job_group_require[k]+1)

    #     for k in range(max_weight_spine):
    #         m.addConstr(gurobipy.quicksum( r[j] for j in range(len(spine_state_vec)) if int(k) == int(j%max_weight_spine)) == job_running_group_require[k])

    #     for it in range(len(spine_state_vec)):
    #         m.addConstr(obj_x_i[it]>=(x_i[it]-spine_state_vec[it])*math.pow(2,it%max_weight_spine))
    #         m.addConstr(obj_x_i[it]>=(spine_state_vec[it]-x_i[it])*math.pow(2,it%max_weight_spine))
    #         m.addConstr(obj_x_i[it]>=(r[it] - running_group_list[it])*math.pow(2,it%max_weight_spine))
    #         m.addConstr(obj_x_i[it]>=(running_group_list[it]-r[it])*math.pow(2,it%max_weight_spine))
    #     m.addConstr(gurobipy.quicksum( obj_x_i[j] for j in range(len(spine_state_vec))) == obj_val)
    #     m.addConstr(gurobipy.quicksum((r[j] - running_group_list[j])*math.pow(2,j%max_weight_spine) for j in range(len(spine_state_vec)) )==0)
    #     # 开始执行
    #     m.update()
    #     m.optimize()
    #     # 记录运行结果
    #     if m.status == gurobipy.GRB.Status.OPTIMAL:
    #         print(int(obj_val.X))
    #         x_i_solution = m.getAttr('X',x_i)
    #         spine_total_group_map = {}
    #         temp_spine_ptr = 0
    #         temp_total_group_list = []
    #         for x_index in x_i_solution:
    #             if x_index%max_weight_spine == 0 and x_index>0:
    #                 spine_total_group_map[temp_spine_ptr] = temp_total_group_list
    #                 temp_total_group_list = []
    #                 temp_spine_ptr+=1
    #             temp_total_group_list.append(int(x_i_solution[x_index]*math.pow(2,int(x_index%max_weight_spine))))
    #         spine_total_group_map[temp_spine_ptr] = temp_total_group_list
    #         print(spine_total_group_map)

    #         r_solution = m.getAttr('X',r)
    #         spine_total_running_group_map = {}
    #         temp_spine_ptr = 0
    #         temp_total_group_list = []
    #         for x_index in r_solution:
    #             if x_index%max_weight_spine == 0 and x_index>0:
    #                 spine_total_running_group_map[temp_spine_ptr] = temp_total_group_list
    #                 temp_total_group_list = []
    #                 temp_spine_ptr+=1
    #             temp_total_group_list.append(int(r_solution[x_index]*math.pow(2,int(x_index%max_weight_spine))))
    #         spine_total_running_group_map[temp_spine_ptr] = temp_total_group_list
    #         print(spine_total_running_group_map)

    #         print("spine group migration map")
    #         spine_task_migration_out_pair = []
    #         spine_task_migration_in_pair = []
    #         for it in range(len(running_group_list)):
    #             if int(r_solution[it])!=running_group_list[it]:
    #                 temp_spine_id = int(it/max_weight_spine)
    #                 temp_class_id = int(it%max_weight_spine)
    #                 if(int( pow(2,temp_class_id)*(r_solution[it] - running_group_list[it]))<0):
    #                     spine_task_migration_out_pair.append([temp_spine_id,int( -1*pow(2,temp_class_id)*(r_solution[it] - running_group_list[it])) ])
    #                 else:
    #                     spine_task_migration_in_pair.append([temp_spine_id,int( pow(2,temp_class_id)*(r_solution[it] - running_group_list[it])) ])

    #         group_migration_map = {}
    #         spine_task_migration_out_pair.sort(key=lambda x: x[1])
    #         spine_task_migration_in_pair.sort(key=lambda x: x[1])
    #         print(spine_task_migration_out_pair)
    #         print(spine_task_migration_in_pair)
    #         out_ptr_left = 0
    #         out_ptr_right = 0
    #         in_ptr = 0
    #         while(out_ptr_right<len(spine_task_migration_out_pair) and in_ptr<len(spine_task_migration_in_pair)):
    #             temp_match_flag = False
    #             temp_sum_group_num = 0
    #             for i in range(out_ptr_left, out_ptr_right+1, 1):
    #                 temp_sum_group_num += spine_task_migration_out_pair[i][1]
    #             assert temp_sum_group_num<= spine_task_migration_in_pair[in_ptr][1]
    #             if(temp_sum_group_num == spine_task_migration_in_pair[in_ptr][1]):
    #                 temp_match_flag = True
    #             if(temp_match_flag):
    #                 for i in range(out_ptr_left, out_ptr_right+1, 1):
    #                     target_spine = spine_task_migration_out_pair[i][0]
    #                     start_spine = spine_task_migration_in_pair[in_ptr][0]
    #                     group_size = spine_task_migration_out_pair[i][1]
    #                     if start_spine not in group_migration_map:
    #                         group_migration_map[start_spine] = {}
    #                     if target_spine not in group_migration_map[start_spine]:
    #                         group_migration_map[start_spine][target_spine] = []
    #                     group_migration_map[start_spine][target_spine].append(group_size)
                        
    #                     self.spine_list[start_spine].update_spine_group_with_required_num(group_size)
    #                     self.spine_list[target_spine].release_spine_group_with_required_num(group_size)

    #                 out_ptr_left = out_ptr_right+1
    #                 in_ptr += 1
    #             out_ptr_right+=1

    #         print(group_migration_map)
    #         #TODO 是否更新group
    #         return group_migration_map
    #     else:
    #         for spine_id in range(self.spine_num):
    #             print(self.spine_list[spine_id].spine_group)
    #         raise Exception("something wrong5 in gurobi solver")

# 测试用
if __name__ == "__main__":
    server_resource_manager = SpineSwitchManager(2,8)
    print("Step0: ")
    for i in range(2):
        server_resource_manager.spine_list[i].print_resource_info()
    print("Step1: ")
    server_resource_manager.choose_group_in_spine(6,[],[])
    server_resource_manager.choose_group_in_spine(6,[],[])
    for i in range(2):
        server_resource_manager.spine_list[i].print_resource_info()
    print("Step2: ")
    server_resource_manager.choose_group_in_spine(4,[],[])
    for i in range(2):
        server_resource_manager.spine_list[i].print_resource_info()