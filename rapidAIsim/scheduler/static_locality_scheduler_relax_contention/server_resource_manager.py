import math
from collections import Counter

# 控制server资源的调度，包括gpu资源以及gpu group资源，
# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，在leaf_resource_manager中在合适的server中选取合适的
#   group，然后在server_resource_manager中按group大小占用gpu并更新group
# 2. 当需要跨leaf通信时，在server_resource_manager中按locality占用gpu并更新group


class Server:
    def __init__(self, id, gpu_per_server, server_num, leaf_num):
        self.gpu_per_server = gpu_per_server
        self.server_id = id
        self.server_per_leaf = int(server_num/leaf_num)
        
        self.global_gpu_list_in_server = {} # 记录全局编号
        self.gpu_group = [gpu_per_server] #注意group与GPU的更新是否同步
        for gpu_global_index in range(gpu_per_server*id, (1+id)*gpu_per_server, 1):
            self.global_gpu_list_in_server[gpu_global_index] = 0
        
    def remain_gpu_num(self):
        return Counter(self.global_gpu_list_in_server.values())[0]

    # 给定需要的gpu数量，返回离他最近的group的大小,返回的group大于等于required_group_size，
    # 如果没有找到返回-1
    def find_closest_gpu_group_size(self, required_group_size):  
        cloest_group_size = 10000
        for exist_group_size in self.gpu_group:
            # exist_group_size在至少要满足需求的同时要找最小的可行group
            if(exist_group_size >= required_group_size 
               and cloest_group_size > exist_group_size):
                cloest_group_size = exist_group_size
        if cloest_group_size == 10000:
            return [self.server_id, -1]
        return [self.server_id, cloest_group_size]

    # 根据需要的gpu大小，占用gpu并更新group
    def occupy_gpu_with_required_num(self, require_gpu_num):
        have_chosen_gpu_num = int (math.pow( 2, int( math.log2(require_gpu_num) ) ))
        need_group_list = [have_chosen_gpu_num]
        temp_potentional_group_size = have_chosen_gpu_num
        while(have_chosen_gpu_num<require_gpu_num):
            if(have_chosen_gpu_num+int(temp_potentional_group_size) <= require_gpu_num):
                need_group_list.append(temp_potentional_group_size)
                have_chosen_gpu_num += temp_potentional_group_size
                temp_potentional_group_size = int(temp_potentional_group_size/2)
            else:
                temp_potentional_group_size = int(temp_potentional_group_size/2)
        gpu_to_choose_list = []
        for need_group in need_group_list:
            group_to_remove = need_group
            if need_group in self.gpu_group:
                self.gpu_group.remove(need_group)
            else:
                group_to_remove = need_group*2
                group_to_add = [need_group]
                while(group_to_remove not in self.gpu_group):
                    group_to_add.append(group_to_remove)
                    group_to_remove*=2
                self.gpu_group.remove(group_to_remove)
                self.gpu_group.extend(group_to_add)
            has_occupy_num = 0
            gpu_global_index_ptr = self.gpu_per_server*self.server_id
            while(has_occupy_num<need_group):
                assert gpu_global_index_ptr<(1+self.server_id)*self.gpu_per_server
                if(self.global_gpu_list_in_server[gpu_global_index_ptr] == 0):
                    self.global_gpu_list_in_server[gpu_global_index_ptr] = 1
                    has_occupy_num+=1
                    gpu_to_choose_list.append(gpu_global_index_ptr)
                    gpu_global_index_ptr+=1
                else:
                    gpu_global_index_ptr+=1
        return (self.server_id, gpu_to_choose_list)

    # 选定占用的gpu后更新记录的gpu与group资源信息，注意传入的gpulist可能包含不在这个server中的gpu
    def occupy_gpu_with_list(self, gpu_global_index_list):
        use_this_server = False
        require_gpu_num = 0
        for gpu_global_index in gpu_global_index_list:
            if gpu_global_index in self.global_gpu_list_in_server:
                assert self.global_gpu_list_in_server[gpu_global_index] == 0
                self.global_gpu_list_in_server[gpu_global_index] = 1
                use_this_server = True
                require_gpu_num += 1
        if(use_this_server):
            if require_gpu_num in self.gpu_group:
                self.gpu_group.remove(require_gpu_num)
            else:
                group_to_remove = require_gpu_num*2
                group_to_add = [require_gpu_num]
                while(group_to_remove not in self.gpu_group):
                    group_to_add.append(group_to_remove)
                    group_to_remove*=2
                self.gpu_group.remove(group_to_remove)
                self.gpu_group.extend(group_to_add)

    # 选定释放的gpu后更新记录的gpu与group资源信息，注意传入的gpulist可能包含不在这个server中的gpu
    def release_gpu_with_list(self, gpu_global_index_list):
        require_gpu_num = 0
        for gpu_global_index in gpu_global_index_list:
            if gpu_global_index in self.global_gpu_list_in_server:
                assert self.global_gpu_list_in_server[gpu_global_index] == 1
                self.global_gpu_list_in_server[gpu_global_index] = 0
                require_gpu_num = require_gpu_num + 1
        if(require_gpu_num>0):
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
                if release_group not in self.gpu_group:
                    self.gpu_group.append(release_group)
                else:
                    to_del_list = []
                    multi_factor = 1
                    while(multi_factor*release_group in self.gpu_group):
                        to_del_list.append(multi_factor*release_group)
                        multi_factor*=2
                    self.gpu_group.append(multi_factor*release_group)
                    for to_del_group in to_del_list:
                        self.gpu_group.remove(to_del_group)

    # debug用的函数
    def print_resource_info(self):
        print("server id: ",end=" ")
        print(self.server_id)
        print("gpu state: ",end=" ")
        print(self.global_gpu_list_in_server)
        print("group state: ",end=" ")
        print(self.gpu_group)

class ServerResourceManager:
    def __init__(self, server_num, gpu_per_server, leaf_num):
        self.server_num = server_num
        self.gpu_per_server = gpu_per_server
        # 生成server列表
        self.server_list = [] # 禁止排序
        for server_id in range(server_num):
            temp_server = Server(server_id, gpu_per_server, server_num, leaf_num)
            self.server_list.append(temp_server)
        self.server_per_leaf = int(server_num/leaf_num)

    def whether_can_find_valid_server(self,gpu_num):
        require_server_num = math.ceil(gpu_num/self.gpu_per_server)
        require_gpu_num_in_server = min(self.gpu_per_server,gpu_num)
        valid_server_num = 0
        for temp_server in self.server_list:
            if temp_server.remain_gpu_num()>=require_gpu_num_in_server:
                valid_server_num += 1
        if valid_server_num>=require_server_num:
            return True
        else: 
            return False
        
    def choose_gpu_in_one_leaf_eleminating_fragmentation(self, leaf_id, need_group_size):
        gpu_to_choose_list = []
        # 先找到相关的server
        relate_server_list = []
        for server_id in range(self.server_num):
            if int(server_id/self.server_per_leaf) == leaf_id and sum(self.server_list[server_id].gpu_group)>0:
                relate_server_list.append(self.server_list[server_id])
        relate_server_list.sort(key=lambda x: sum(x.gpu_group))
        remain_to_choose_gpu = need_group_size
        for server_ in relate_server_list:
            if remain_to_choose_gpu<0:
                print("something wrong in choose_gpu_in_one_leaf_eleminating_fragmentation")
            if remain_to_choose_gpu==0:
                break
            to_chosen_num = min(remain_to_choose_gpu, sum(server_.gpu_group))
            chosen_gpu = server_.occupy_gpu_with_required_num(to_chosen_num)[1]
            gpu_to_choose_list.extend(chosen_gpu)
            remain_to_choose_gpu -= to_chosen_num
        return gpu_to_choose_list
    
    # 当无需跨leaf通信时，根据给定leaf spine需要的group size，在对于的server中选择gpu
    def choose_gpu_in_one_leaf(self, leaf_id, need_group_size):
        print("debug leaf_id, need_group_size",leaf_id, need_group_size)
        gpu_to_choose_list = []
        # 先找到相关的server
        relate_server_list = []
        for server_id in range(self.server_num):
            if int(server_id/self.server_per_leaf) == leaf_id:
                relate_server_list.append(self.server_list[server_id])
        # 按如下逻辑占用gpu：
        # 首先尝试独占，然后尝试在两个server中平均占用，然后尝试在四个server中平均占用
        can_deuce = False
        need_server_num = max(1, int(need_group_size/self.gpu_per_server))
        while(need_server_num<=self.server_per_leaf):
            potentional_server_groupsize_pair_list = []
            need_group_each_server = int(need_group_size/need_server_num)
            for potentional_server in relate_server_list:
                closest_group_size = potentional_server.find_closest_gpu_group_size(need_group_each_server)[1]
                if(closest_group_size>=need_group_each_server):
                    potentional_server_groupsize_pair_list.append([potentional_server, closest_group_size])
            if(len(potentional_server_groupsize_pair_list) < need_server_num):
                need_server_num *= 2
            else:
                # 将列表按选择的group大小从小到大排列，然后选择前need_server_num个server进行占用
                potentional_server_groupsize_pair_list.sort(key=lambda x: x[1])
                need_group_each_server = int(need_group_size/need_server_num)
                for pair_index in range(need_server_num):
                    chosen_gpu = potentional_server_groupsize_pair_list[pair_index][0].occupy_gpu_with_required_num(need_group_each_server)[1]
                    gpu_to_choose_list.extend(chosen_gpu)
                can_deuce = True
                break
        # 如果都不行，那么至少要跨server通信，尽可能占用少的server,不要求平分，比如需要个gpu，那么可以选择6，2这样的组合
        if not can_deuce:
            # 首先记录(server_id, group_size),按group size从大到小排列，因为need_group_size肯定大于最大的group_size，因此可以推断出所需的group大小
            serverid_groupsize_pair_list = []
            for potentional_server in relate_server_list:
                for potention_group_size in potentional_server.gpu_group:
                    serverid_groupsize_pair_list.append([potentional_server.server_id, potention_group_size])
            serverid_groupsize_pair_list.sort(key=lambda x: x[1], reverse=True)
            chosen_gpu_num = 0
            # 一开始期望的group大小为所有server中最大的group的大小, 不断选择group直至满足条件
            while(chosen_gpu_num<need_group_size):
                if(chosen_gpu_num + serverid_groupsize_pair_list[0][1]>need_group_size):
                    print("something wroing in can_deuce")
                    exit()
                    del serverid_groupsize_pair_list[0]
                temp_chosen_gpu_list = self.server_list[serverid_groupsize_pair_list[0][0]].occupy_gpu_with_required_num(serverid_groupsize_pair_list[0][1])[1]
                gpu_to_choose_list.extend(temp_chosen_gpu_list)
                chosen_gpu_num += serverid_groupsize_pair_list[0][1]
                del serverid_groupsize_pair_list[0]
            assert chosen_gpu_num == need_group_size
        # 首先记录(server_id, group_size),按group size从大到小排列，因为need_group_size肯定大于最大的group_size，因此可以推断出所需的group大小
        # serverid_groupsize_pair_list = []
        # for potentional_server in relate_server_list:
        #     for potention_group_size in potentional_server.gpu_group:
        #         serverid_groupsize_pair_list.append([potentional_server.server_id, potention_group_size])
        # serverid_groupsize_pair_list.sort(key=lambda x: x[1], reverse=True)
        # chosen_gpu_num = 0
        # # 一开始期望的group大小为所有server中最大的group的大小, 不断选择group直至满足条件
        # while(chosen_gpu_num<need_group_size):
        #     # if(chosen_gpu_num + serverid_groupsize_pair_list[0][1]>need_group_size):
        #     #     print("something wroing in can_deuce")
        #     #     exit()
        #     #     del serverid_groupsize_pair_list[0]
        #     need_group_each_server = min(need_group_size-chosen_gpu_num,serverid_groupsize_pair_list[0][1])
        #     temp_chosen_gpu_list = self.server_list[serverid_groupsize_pair_list[0][0]].occupy_gpu_with_required_num(need_group_each_server)[1]
        #     gpu_to_choose_list.extend(temp_chosen_gpu_list)
        #     chosen_gpu_num += need_group_each_server
        #     del serverid_groupsize_pair_list[0]
        # assert chosen_gpu_num == need_group_size


        return gpu_to_choose_list



    # 当需要跨leaf通信时，则完全依靠locality选择GPU，因为通过网络重构肯定可以将通信需求迁移到同一个spine上
    # 注意此时返回占用的gpu，调用方还要根据占用的gpu反推出leaf group的占用情况
    def choose_gpu_with_locality(self, need_group_size):
        gpu_to_choose_list = []
        # 按如下逻辑占用gpu：
        # 首先尝试独占，然后尝试在两个server中平均占用，然后尝试在四个server中平均占用,依次类推
        can_deuce = False
        need_server_num = 1
        while(need_server_num<=self.server_num):
            potentional_server_groupsize_pair_list = []
            need_group_each_server = int(need_group_size/need_server_num)
            for potentional_server in self.server_list:
                closest_group_size = potentional_server.find_closest_gpu_group_size(need_group_each_server)[1]
                if(closest_group_size>=need_group_each_server):
                    potentional_server_groupsize_pair_list.append([potentional_server, closest_group_size])
            if(len(potentional_server_groupsize_pair_list) < need_server_num):
                need_server_num *= 2
            else:
                # 当找到足够多的valid server，将列表按选择的group大小从小到大排列，然后选择前need_server_num个server进行占用
                potentional_server_groupsize_pair_list.sort(key=lambda x: x[1])
                need_group_each_server = int(need_group_size/need_server_num)
                for pair_index in range(need_server_num):
                    chosen_gpu = potentional_server_groupsize_pair_list[pair_index][0].occupy_gpu_with_required_num(need_group_each_server)[1]
                    gpu_to_choose_list.extend(chosen_gpu)
                can_deuce = True
                break
        # 如果都不行，那么至少要跨server通信，尽可能占用少的server,不要求平分，比如需要个gpu，那么可以选择6，2这样的组合
        if not can_deuce:
            # 首先记录(server_id, group_size),按group size从大到小排列，因为need_group_size肯定大于最大的group_size，因此可以推断出所需的group大小
            serverid_groupsize_pair_list = []
            for potentional_server in self.server_list:
                for potention_group_size in potentional_server.gpu_group:
                    serverid_groupsize_pair_list.append([potentional_server.server_id, potention_group_size])
            serverid_groupsize_pair_list.sort(key=lambda x: x[1], reverse=True)
            chosen_gpu_num = 0
            # 一开始期望的group大小为所有server中最大的group的大小, 不断选择group直至满足条件
            while(chosen_gpu_num<need_group_size):
                assert(len(serverid_groupsize_pair_list)>0)
                if(chosen_gpu_num + serverid_groupsize_pair_list[0][1]>need_group_size):
                    print("something wroing in can_deuce")
                    exit()
                    del serverid_groupsize_pair_list[0]
                temp_chosen_gpu_list = self.server_list[serverid_groupsize_pair_list[0][0]].occupy_gpu_with_required_num(serverid_groupsize_pair_list[0][1])[1]
                gpu_to_choose_list.extend(temp_chosen_gpu_list)
                chosen_gpu_num += serverid_groupsize_pair_list[0][1]
                del serverid_groupsize_pair_list[0]
            assert chosen_gpu_num == need_group_size

        return gpu_to_choose_list

    def release_gpu_in_server(self, gpu_to_release_list):
        # 释放server中的GPU资源
        for temp_server in self.server_list:
            temp_server.release_gpu_with_list(gpu_to_release_list)

    def print_info(self):
        for temp_server in self.server_list:
            temp_server.print_resource_info()

    def cal_remain_gpu_num(self):
        remain_gpu_n = 0
        for server in self.server_list:
            remain_gpu_n += server.remain_gpu_num()
        return remain_gpu_n

    def return_server_remain_gpuNum_map(self):
        server_remain_gpuNum_map = {}
        for server in self.server_list:
            server_remain_gpuNum_map[server.server_id] = server.remain_gpu_num()
        return server_remain_gpuNum_map

# 测试用
if __name__ == "__main__":
    server_resource_manager = ServerResourceManager(4,8)
    gpu_to_use_list = []
    gpu_to_use_list.extend(server_resource_manager.choose_gpu_with_locality(6))
    print("First Step:")
    server_resource_manager.print_info()
    gpu_to_use_list.extend(server_resource_manager.choose_gpu_with_locality(8))
    print("Second Step:")
    server_resource_manager.print_info()
    server_resource_manager.release_gpu_in_server(gpu_to_use_list)
    print("Third Step:")
    server_resource_manager.print_info()
