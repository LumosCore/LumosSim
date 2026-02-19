import rapidAIsim.scheduler.hw_new_arg_sim.server
import rapidAIsim.scheduler.hw_new_arg_sim.job as job
import rapidAIsim.scheduler.hw_new_arg_sim.oxc_group as oxc_group
import rapidAIsim.scheduler.hw_new_arg_sim.ele_group as ele_group
import gurobipy


class NaiveScheduler:
    def __init__(self, tor_num=512, ele_group_num=8, oxc_group_num=4):
        self.total_tor_num = tor_num
        self.ele_group_num = ele_group_num
        self.oxc_group_num = oxc_group_num
        self.colum_per_oxc = int(tor_num/self.oxc_group_num/self.ele_group_num )
        self.oxc_list = [oxc_group.OXC_Group(i,int(tor_num/self.oxc_group_num)) for i in range(self.oxc_group_num)]
        self.ele_list = [ele_group.ELE_Group(i,int(tor_num/self.ele_group_num)) for i in range(self.ele_group_num)]
        self.job_id_map= {}
        
    def guroby_solver(self, generated_matrix, v, u, group_coff):
        m = gurobipy.Model("Clos solution")
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 120)
        m.setParam('ConcurrentMIP', 64)
        a_total_num = len(generated_matrix)
        b_total_num = self.oxc_group_num
        name_list_y_i_j = []
        name_list_a_i = []
        name_list_b_j = []
        for i in range(a_total_num):
            name_list_a_i.append(str(i))
        for j in range(b_total_num):
            name_list_b_j.append(str(j))
        for i in range(a_total_num):
            for j in range(b_total_num):
                name_list_y_i_j.append(str(i)+"_"+str(j))
                
        y_i_j = {}
        c_i_j = {}
        a_i = {}
        b_j = {}
        print("start:")
        for i in range(a_total_num):
            for j in range(b_total_num):
                y_i_j[str(i)+"_"+str(j)] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=generated_matrix[i][j],name='y_i_j')
                c_i_j[str(i)+"_"+str(j)] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=-1, ub=0,name='c_i_j')
                print(generated_matrix[i][j],end=" ")
            print()
        for it in name_list_a_i:
            a_i[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=1,name='a_i')
        for it in name_list_b_j:
            b_j[it] = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.colum_per_oxc,name='b_j')
        max_b = m.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=self.colum_per_oxc,name='b_j')
        obj_val = m.addVar(lb=-10000, ub=100000,name='obj')
        m.setObjective(obj_val, gurobipy.GRB.MINIMIZE)
            
        # 线性化变量
        for i in range(a_total_num):
            for j in range(b_total_num):
                m.addConstr(y_i_j[str(i)+"_"+str(j)]<=u*a_i[str(i)])
                m.addConstr(y_i_j[str(i)+"_"+str(j)]<=b_j[str(j)])
                m.addConstr(y_i_j[str(i)+"_"+str(j)]>=b_j[str(j)] - u*(1-a_i[str(i)]))
                m.addConstr(c_i_j[str(i)+"_"+str(j)]<=-0.001*y_i_j[str(i)+"_"+str(j)])
        
        # 横
        for i in range(a_total_num):
            m.addConstr(gurobipy.quicksum( y_i_j[str(i)+'_'+str(j)] for j in range(b_total_num) ) == u*a_i[str(i)])
                
        # 竖
        m.addConstr(gurobipy.quicksum( a_i[str(i)] for i in range(a_total_num) ) == v)
        for j in range(b_total_num):
            max_b >= b_j[str(j)]
        m.addConstr(obj_val ==  gurobipy.quicksum( a_i[str(i)]*group_coff[i] for i in range(a_total_num) ))
        
        #m.addConstr(obj_val ==  gurobipy.quicksum( c_i_j[str(i)+'_'+str(j)] for j in range(b_total_num) for i in range(a_total_num) ))
        
        
        
        # 开始执行
        m.update()
        m.optimize()
        # 记录运行结果
        if m.status == gurobipy.GRB.Status.OPTIMAL:
            print(int(obj_val.X))
            valid= True
            a_i_solution = m.getAttr('X',a_i)
            b_j_solution = m.getAttr('X',b_j)
            y_solution = m.getAttr('X',y_i_j)
            a_res = [round(a_i_solution[str(it)]) for it in range(a_total_num)]
            b_res = [round(b_j_solution[str(it)]) for it in range(b_total_num)]
            y_res = [[round(y_solution[str(i)+"_"+str(j)]) for j in range(b_total_num)] for i in range(a_total_num)]
            # print(a_res)
            # print(b_res)
            print("res:")
            for i in range(a_total_num):
                for j in range(b_total_num):
                    print(y_res[i][j],end=" ")
                print()
            # print(y_solution)
            print()
            return True, a_res, b_res, y_res
        
        else:
            return False, [], [], []
            print("no solution")
            
        
    # 首先根据gpu数量选择gpu形状，为了尽可能分配到同一个group,x要尽可能小
    def decide_shape(self, used_tor_num):
        x = max(1,int(used_tor_num/int(self.total_tor_num/self.ele_group_num)))
        y = int(used_tor_num/x)
        allocate_success = False
        while not allocate_success and x <= self.ele_group_num and y >= 1:
            potention_valid_ele_list = []
            for tmp_ele in self.ele_list:
                if tmp_ele.remain_tor_in_this_group >= y:
                    potention_valid_ele_list.append(tmp_ele)
            if len(potention_valid_ele_list) >= x:
                generated_matrix = [[0 for i in range(self.oxc_group_num)] for j in range(len(potention_valid_ele_list))] # 根据潜在group生成matrix
                tmp_id_ele_id_map = {}
                for potentional_ele_id in range(len(potention_valid_ele_list)):
                    potentional_ele = potention_valid_ele_list[potentional_ele_id]
                    tmp_id_ele_id_map[potentional_ele_id] =potentional_ele.id
                    for tmp_tor_id in range(len(potentional_ele.tor_list)):
                        if potentional_ele.tor_list[tmp_tor_id] == 0:
                            tmp_oxc_id = int(tmp_tor_id/self.colum_per_oxc)
                            generated_matrix[potentional_ele_id][tmp_oxc_id] += 1
                group_coff = [ele.remain_tor_in_this_group for ele in potention_valid_ele_list]
                is_valid, a_res, b_res, y_res = self.guroby_solver(generated_matrix,x,y,group_coff)
                if(is_valid):
                    tor_occupy_list = []
                    for potentional_ele_id in range(len(y_res)):
                        ele_id = tmp_id_ele_id_map[potentional_ele_id]
                        for oxc_id in range(len(y_res[potentional_ele_id])):
                            if y_res[potentional_ele_id][oxc_id] > 0:
                                tor_occupy_list.extend(self.ele_list[ele_id].occupy_tor(oxc_id, self.colum_per_oxc, y_res[potentional_ele_id][oxc_id]))
                    allocate_success = True
                    return True,tor_occupy_list
            x *= 2
            y = int(used_tor_num/x)
        return False,[]
    
    
    def try_to_allocate_job(self, cur_job):
        allocate_success, tor_occupy_list = self.decide_shape(cur_job.used_tor_num)
        if allocate_success:
            cur_job.allocated_tors = tor_occupy_list
            return True, tor_occupy_list
        return False, []
    
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
        allocated_link_mapping = []
        cur_job = job.Job(job_id, used_tor_num)
        allocate_success, tor_occupy_list = self.try_to_allocate_job(cur_job)
        if allocate_success:
            cur_job.start_time = sim_time
            self.job_id_map[job_id] = cur_job
        else:
            if free_gpu>used_tor_num:
                print("no resource ",used_tor_num, free_gpu)
        f2 = open('queue_length.txt','a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()
        return allocate_success, tor_occupy_list, allocated_link_mapping, None, None
    
    
# # deubg       
# test_matrix = [[2,2,0,0],
#                [1,2,2,2],
#                [1,2,0,1],
#                [0,2,2,1],]
# test_matrix = [[1,2,2,1],
#                [1,2,0,2],
#                [1,2,1,2],
#                [1,0,0,2],]
# naive_sch = NaiveScheduler(32,4,4)
# naive_sch.guroby_solver(test_matrix,4,2)