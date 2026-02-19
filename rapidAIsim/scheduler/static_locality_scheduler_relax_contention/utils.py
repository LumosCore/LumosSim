
import gurobipy
import math
import numpy as np

def power_of_2(n):
    start = 1
    flag = False
    while start <= n:
        if start == n: flag = True
        start *= 2
    return flag

def power2_zero(n):
    if n == 0: return True
    start = 1
    flag = False
    while start <= n:
        if start == n: flag = True
        start *= 2
    return flag

def chose_server_map(gpu_index_list, server_size = 8):
    server_num_map = {}
    for gpu_id in gpu_index_list:
        server_id = int(gpu_id/8)
        if server_id not in server_num_map:
            server_num_map[server_id] = 0
        server_num_map[server_id] += 1
    return server_num_map

def find_first_element_list(pair_list, second_value):
    result = []
    for temp_pair in pair_list:
        if temp_pair[1] == second_value:
            result.append(temp_pair[0])
    return result

def find_closest_first_list_pair(pair_list, second_value):
    result_index = -1
    cur_gap = 10000000
    for temp_pair in pair_list:
        if temp_pair[1]-second_value>=0 and temp_pair[1]-second_value<cur_gap:
            cur_gap = temp_pair[1]-second_value
            result_index = temp_pair[0]
    assert result_index != -1
    assert cur_gap != 0
    return [result_index, second_value+cur_gap]

def get_leaf_module_id(leaf_id, gpu_num):
        return leaf_id+gpu_num

def get_spine_module_id(spine_id, gpu_num, leaf_num):
    return spine_id+gpu_num+leaf_num

    
            # while(len(choosed_spine_group_pair_list)>0):
            #     assert choosed_spine_group_pair_list[0][1] == choosed_spine_group_pair_list[1][1]
            #     final_target_spine_index = choosed_spine_group_pair_list[1][0]
            #     if choosed_spine_group_pair_list[0][0] not in group_migration_map:
            #         group_migration_map[choosed_spine_group_pair_list[0][0]] = {}
            #     if choosed_spine_group_pair_list[1][0] not in group_migration_map[choosed_spine_group_pair_list[0][0]]:
            #         group_migration_map[choosed_spine_group_pair_list[0][0]][choosed_spine_group_pair_list[1][0]] = []
            #     group_migration_map[choosed_spine_group_pair_list[0][0]][choosed_spine_group_pair_list[1][0]].append(choosed_spine_group_pair_list[0][1])
            #     del choosed_spine_group_pair_list[0]
            #     # 每一次都选取两个最小的group进行合并，即将其中一个spine中大小之和为n的任务迁移到另一个spine中大小为n的group中运行
            #     choosed_spine_group_pair_list.sort(key=lambda x: x[1],reverse=False)

def new_make_spine_migration_strategy():
        print("start migration")
        init_spine_group_list = [[0,0,0,0,1,1,0], [0,0,0,1,1,1,0]] # [16,32],[16] # [8,16,32],[8]
        job_running_group_require = [0,0,0,1,1,0,0]
        job_group_require = [0,0,0,1,1,0,1]

        spine_state_vec = []
        for temp_spine_state_vec in init_spine_group_list:
            spine_state_vec.extend(temp_spine_state_vec)
        running_group_list = []
        for temp_spine_group in init_spine_group_list:
            temp_running_group = [0 for i in range(7)]
            remain_gpu = 64
            for temp_element in range(len(temp_spine_group)):
                if(temp_spine_group[temp_element]):
                    remain_gpu -= int(pow(2,temp_element))
            have_chosen_group_size = int (math.pow( 2, int( math.log2(remain_gpu) ) ))
            temp_running_group[int(math.log2(have_chosen_group_size))] = 1
            temp_potentional_group_size = have_chosen_group_size
            while(have_chosen_group_size<remain_gpu):
                if(have_chosen_group_size+int(temp_potentional_group_size) <= remain_gpu):
                    temp_running_group[int( math.log2(temp_potentional_group_size))] = 1
                    have_chosen_group_size += temp_potentional_group_size
                    temp_potentional_group_size = int(temp_potentional_group_size/2)
                else:
                    temp_potentional_group_size = int(temp_potentional_group_size/2)
            running_group_list.extend(temp_running_group)
        print(running_group_list)
            
        m = gurobipy.Model("Clos solution")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 120)

        x_i = {}
        for it in range(len(spine_state_vec)):
            x_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='x_i')
        r = {}
        for it in range(len(spine_state_vec)):
            r[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=64,name='r_i')

        obj_val = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=4096,name='obj')
        obj_x_i = {}
        for it in range(len(spine_state_vec)):
            obj_x_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='obj_x_i')
        obj_r_i = {}
        for it in range(len(spine_state_vec)):
            obj_r_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='obj_r_i')
        m.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
        m.update()

        # 设置条件
        for it in range(len(spine_state_vec)):
            m.addConstr(gurobipy.quicksum( (x_i[j] + r[j])*math.pow(2,int(j%7)) for j in range(len(spine_state_vec)) if int(it/7) == int(j/7)) == 64)

        for k in range(7):
            temp_class_list = []
            for temp_k in range(k,7,1):
                temp_class_list.append(temp_k)
            temp_value = 0
            for temp_k in range(k,7,1):
                temp_value += int(job_group_require[k]*pow(2,temp_k))
            m.addConstr(gurobipy.quicksum( (x_i[j] + r[j])*math.pow(2,int(j%7)) for j in range(len(spine_state_vec)) if int(j%7) in temp_class_list) >= temp_value)

        for k in range(6):
            m.addConstr(gurobipy.quicksum( (x_i[j] + r[j]) for j in range(len(spine_state_vec)) if int(k) == int(j%7)) <= job_group_require[k]+1)

        for k in range(7):
            m.addConstr(gurobipy.quicksum( r[j] for j in range(len(spine_state_vec)) if int(k) == int(j%7)) == job_running_group_require[k])

        for it in range(len(spine_state_vec)):
            m.addConstr(obj_x_i[it]>=x_i[it]-spine_state_vec[it])
            m.addConstr(obj_x_i[it]>=spine_state_vec[it]-x_i[it])
        m.addConstr(gurobipy.quicksum( obj_x_i[j] for j in range(len(spine_state_vec))) == obj_val)

        # 开始执行
        m.update()
        m.optimize()
        # 记录运行结果
        if m.status == gurobipy.GRB.Status.OPTIMAL:
            print(int(obj_val.X))
            x_i_solution = m.getAttr('X',x_i)
            spine_total_group_map = {}
            temp_spine_ptr = 0
            temp_total_group_list = []
            for x_index in x_i_solution:
                if x_index%7 == 0 and x_index>0:
                    spine_total_group_map[temp_spine_ptr] = temp_total_group_list
                    temp_total_group_list = []
                    temp_spine_ptr+=1
                temp_total_group_list.append(int(x_i_solution[x_index]*math.pow(2,int(x_index%7))))
            spine_total_group_map[temp_spine_ptr] = temp_total_group_list
            print(spine_total_group_map)

            r_solution = m.getAttr('X',r)
            spine_total_running_group_map = {}
            temp_spine_ptr = 0
            temp_total_group_list = []
            for x_index in r_solution:
                if x_index%7 == 0 and x_index>0:
                    spine_total_running_group_map[temp_spine_ptr] = temp_total_group_list
                    temp_total_group_list = []
                    temp_spine_ptr+=1
                temp_total_group_list.append(int(r_solution[x_index]*math.pow(2,int(x_index%7))))
            spine_total_running_group_map[temp_spine_ptr] = temp_total_group_list
            print(spine_total_running_group_map)

            print("spine group migration map")
            spine_task_migration_map = {}
            spine_task_migration_out_pair = []
            spine_task_migration_in_pair = []
            for it in range(len(running_group_list)):
                if int(r_solution[it])!=running_group_list[it]:
                    temp_spine_id = int(it/7)
                    temp_class_id = int(it%7)
                    if temp_spine_id not in spine_task_migration_map:
                        spine_task_migration_map[temp_spine_id] = []
                    spine_task_migration_map[temp_spine_id].append(int( pow(2,temp_class_id)*(r_solution[it] - running_group_list[it])))
                    if(int( pow(2,temp_class_id)*(r_solution[it] - running_group_list[it]))<0):
                        spine_task_migration_out_pair.append([temp_spine_id,int( -1*pow(2,temp_class_id)*(r_solution[it] - running_group_list[it])) ])
                    else:
                        spine_task_migration_in_pair.append([temp_spine_id,int( pow(2,temp_class_id)*(r_solution[it] - running_group_list[it])) ])

            group_migration_map = {}
            spine_task_migration_out_pair.sort(key=lambda x: x[1],reverse=True)
            spine_task_migration_in_pair.sort(key=lambda x: x[1],reverse=True)
            out_ptr_left = out_ptr_right = 0
            in_ptr = 0
            while(out_ptr_right<len(spine_task_migration_out_pair) and in_ptr<len(spine_task_migration_in_pair)):
                temp_match_flag = False
                temp_sum_group_num = 0
                for i in range(out_ptr_left, out_ptr_right+1, 1):
                    temp_sum_group_num += spine_task_migration_out_pair[i][1]
                assert temp_sum_group_num<= spine_task_migration_in_pair[in_ptr][1]
                if(temp_sum_group_num == spine_task_migration_in_pair[in_ptr][1]):
                    temp_match_flag = True
                if(temp_match_flag):
                    for i in range(out_ptr_left, out_ptr_right+1, 1):
                        start_spine = spine_task_migration_out_pair[i][0]
                        target_spine = spine_task_migration_in_pair[in_ptr][0]
                        group_size = spine_task_migration_out_pair[i][1]
                        if start_spine not in group_migration_map:
                            group_migration_map[start_spine] = {}
                        if target_spine not in group_migration_map[start_spine]:
                            group_migration_map[start_spine][target_spine] = []
                        group_migration_map[start_spine][target_spine].append(group_size)
                    out_ptr_left = out_ptr_right+1
                    in_ptr += 1
                out_ptr_right+=1

            print(spine_task_migration_map)
            print(spine_task_migration_out_pair)
            print(spine_task_migration_in_pair)
            print(group_migration_map)
        else:
            raise Exception("something wrong5 in gurobi solver")
        
if __name__ == "__main__":
    new_make_spine_migration_strategy()
    #  while(temp_require_leaf_num>=1):
    #         temp_require_spine_num = int(gpu_num/temp_require_leaf_num)
    #         # temp_require_gpu_per_leaf = int(gpu_num/temp_require_leaf_num)
    #         # temp_require_port_per_spine = int(gpu_num/temp_require_spine_num)

    #         leaf_remain_empt_server_list = []
    #         for temp_leaf_id in range(self.leaf_num):
    #             leaf_remain_empt_server_list.append(0)
    #         for temp_server_id in range(self.server_num):
    #             temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
    #             if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
    #                 leaf_remain_empt_server_list[temp_leaf_id] += 1
    #         allocate_success, allocation_link_mapping, leaf_occupy_gpu_num_map, spine_occupy_port_num_map, job_allocated_leaf_spine_link = self.connection_manager_.update_leaf_to_spine_map_according_to_gpu_size(gpu_num, leaf_remain_empt_server_list, temp_require_leaf_num, temp_require_spine_num)
    #         if allocate_success:
    #             chosen_gpu_list = []
    #             for chosen_leaf_id in leaf_occupy_gpu_num_map:
    #                 chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf(chosen_leaf_id, leaf_occupy_gpu_num_map[chosen_leaf_id]))
    #                 self.leaf_resource_manager_.leaf_list[chosen_leaf_id].update_leaf_group_with_required_num(leaf_occupy_gpu_num_map[chosen_leaf_id])
    #             for chosen_spine_id in spine_occupy_port_num_map:
    #                 self.spine_resource_manager_.spine_list[chosen_spine_id].update_spine_group_with_required_num(spine_occupy_port_num_map[chosen_spine_id])
    #             for output_gpu_index in chosen_gpu_list:
    #                 output_leaf_index = utils.get_leaf_module_id(chosen_leaf_id, self.gpu_num)
    #                 allocation_link_mapping.append([output_gpu_index, output_leaf_index, 1])
    #                 allocation_link_mapping.append([output_leaf_index, output_gpu_index, 1])
    #             new_job.start_time = sim_time
    #             new_job.allocated_gpus = chosen_gpu_list
    #             new_job.job_allocated_leaf_spine_link = job_allocated_leaf_spine_link
    #             self.current_job_list[job_id] = new_job
    #             print("finish allocation2")
