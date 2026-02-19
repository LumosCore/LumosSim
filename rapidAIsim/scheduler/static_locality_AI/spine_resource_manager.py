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
        assert sum(self.spine_group)>=0
        return self.spine_id

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
    def __init__(self, spine_num = 8, port_per_spine = 32, banned_spine_list = []):
        self.spine_num = spine_num
        self.port_per_spine = port_per_spine
        # 生成spine列表
        self.spine_list = [] # 禁止排序
        for spine_id in range(spine_num):
            temp_spine = SpineSwitch(spine_id, port_per_spine)
            self.spine_list.append(temp_spine)
        self.banned_spine_list = banned_spine_list

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

    def num_valid_spine(self, need_port_num):
        valid_num = 0
        for spine in self.spine_list:
            if sum(spine.spine_group) >= need_port_num:
                valid_num += 1
        return valid_num


    # 根据需要的端口数量判断是否存在合适的spine group，如果是则返回true，spine id并更新group（具体端口选择
    # 在connection manager中进行），如果不存在则返回false以及spine迁移方案
    # def choose_group_in_spine(self, require_gpu_num):
    #     need_group_size = min(self.port_per_spine, require_gpu_num)
    #     need_spine_num = int(require_gpu_num/need_group_size)
    #     potential_spineId_group_pair_list = []
    #     for temp_spine in self.spine_list:
    #         if temp_spine.spine_id not in self.banned_spine_list:
    #             group_in_this_spine = temp_spine.find_closest_spine_group_size(need_group_size)
    #             if group_in_this_spine[1]!=-1:
    #                 potential_spineId_group_pair_list.append(group_in_this_spine)
    #     # 如果有合适的spine交换机,那么根据spine交换机信息按选择的group大小从小到大排序，
    #     # 选择合适的spine并更新group
    #     if(len(potential_spineId_group_pair_list)>=need_spine_num):
    #         potential_spineId_group_pair_list.sort( key=lambda x: (x[1]) ) # 选择最小的符合条件的group
    #         choosed_spine_index_list = []
    #         for have_chosen_spine_num in range(need_spine_num):
    #             choosed_spine = self.spine_list[potential_spineId_group_pair_list[have_chosen_spine_num][0]]
    #             choosed_spine.update_spine_group_with_required_num(need_group_size)
    #             choosed_spine_index_list.append(choosed_spine.spine_id)
    #         return True, choosed_spine_index_list
    #     else:
    #         return False, None, None
    def choose_group_in_spine(self, require_gpu_num):
        #if require_gpu_num == pow(2,math.ceil(math.log2(require_gpu_num))):
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


