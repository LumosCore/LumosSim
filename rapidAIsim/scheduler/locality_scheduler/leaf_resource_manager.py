import copy
import math
from collections import Counter

# 控制leaf资源的调度，包括group
# gpu调度分两个阶段：
# 1. 当能够不跨leaf通信时，在leaf_resource_manager中选取合适的
#   group并更新group信息
# 2. 当需要跨leaf通信时，在server_resource_manager中按locality占用gpu，将gpu
# 信息转化为leaf group的更新信息，然后按数量更新相关leaf的group

class LeafSwitch:
    def __init__(self, id, gpu_per_leaf):
        self.gpu_port_per_leaf = gpu_per_leaf
        self.leaf_id = id
        self.leaf_group = [gpu_per_leaf] #注意group与GPU的更新是否同步

    # 给定需要的gpu数量，返回离他最近的group的大小,返回的group大于等于required_group_size，
    # 如果没有找到返回-1
    def find_closest_leaf_group_size(self, required_group_size):  
        cloest_group_size = 10000
        for exist_group_size in self.leaf_group:
            # exist_group_size在至少要满足需求的同时要找最小的可行group
            if(exist_group_size >= required_group_size 
               and cloest_group_size > exist_group_size):
                cloest_group_size = exist_group_size
        if cloest_group_size == 10000:
            return [self.leaf_id, -1]
        return [self.leaf_id, cloest_group_size]


    # 根据需要的gpu大小，更新group
    def update_leaf_group_with_required_num(self, require_gpu_num):
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
        for need_group in need_group_list:
            group_to_remove = need_group
            if need_group in self.leaf_group:
                self.leaf_group.remove(need_group)
            else:
                group_to_remove = need_group*2
                group_to_add = [need_group]
                while(group_to_remove not in self.leaf_group):
                    group_to_add.append(group_to_remove)
                    group_to_remove*=2
                self.leaf_group.remove(group_to_remove)
                self.leaf_group.extend(group_to_add)
        return self.leaf_id


    # 根据group大小释放资源，更新leaf group
    def release_leaf_group_with_required_num(self, require_gpu_num):
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
            if release_group not in self.leaf_group:
                self.leaf_group.append(release_group)
            else:
                to_del_list = []
                multi_factor = 1
                while(multi_factor*release_group in self.leaf_group):
                    to_del_list.append(multi_factor*release_group)
                    multi_factor*=2
                self.leaf_group.append(multi_factor*release_group)
                for to_del_group in to_del_list:
                    self.leaf_group.remove(to_del_group)

    # debug用的函数
    def print_resource_info(self):
        print("leaf id: ",end=" ")
        print(self.leaf_id)
        print("group state: ",end=" ")
        print(self.leaf_group)
    
class LeafResourceManager:
    def __init__(self, leaf_num = 16, gpu_per_leaf = 32):
        self.leaf_num = leaf_num
        self.gpu_per_leaf = gpu_per_leaf
        # 生成leaf列表
        self.leaf_list = [] # 禁止排序
        for leaf_id in range(leaf_num):
            temp_leaf = LeafSwitch(leaf_id, gpu_per_leaf)
            self.leaf_list.append(temp_leaf)

    def print_remain_leaf_port_num(self):
        print("{",end="")
        for leaf in self.leaf_list:
            if leaf.leaf_group!=[]:
                print(leaf.leaf_id,end=": ")
                print(sum(leaf.leaf_group),end=", ")
        print()

    # 根据需要的GPU数量判断是否跨leaf通信，如果不跨则返回true以及选取的leaf并更新group，如果要跨则返回false
    def choose_gpu_in_leaf(self, require_gpu_num):
        potential_leafId_group_pair_list = []
        for temp_leaf in self.leaf_list:
            group_in_this_leaf = temp_leaf.find_closest_leaf_group_size(require_gpu_num)
            if group_in_this_leaf[1]!=-1:
                potential_leafId_group_pair_list.append(group_in_this_leaf)
        # 如果有合适的leaf交换机,那么根据leaf交换机信息选择合适的leaf并更新group
        if(len(potential_leafId_group_pair_list)>0):
            potential_leafId_group_pair_list.sort( key=lambda x: (x[1]) ) # 选择最小的符合条件的group
            choosed_leaf = self.leaf_list[potential_leafId_group_pair_list[0][0]]
            choosed_leaf.update_leaf_group_with_required_num(require_gpu_num)
            return True, choosed_leaf.leaf_id
        else:
            return False, True

    # 在需要跨leaf通信的情况下，传入选取的gpu列表，然后更新对应leaf的group
    def update_group_with_given_gpu_list(self, chosen_global_gpu_index_list):
        chosen_leaf_id_num_list = []
        chosen_leaf_group_dict = {}
        for temp_global_gpu_index in chosen_global_gpu_index_list:
            temp_leaf_id = int(temp_global_gpu_index/self.gpu_per_leaf)
            if temp_leaf_id not in chosen_leaf_group_dict:
                chosen_leaf_group_dict[temp_leaf_id] = 0
            chosen_leaf_group_dict[temp_leaf_id] += 1
        for temp_leaf_id in chosen_leaf_group_dict:
            self.leaf_list[temp_leaf_id].update_leaf_group_with_required_num(chosen_leaf_group_dict[temp_leaf_id])
            chosen_leaf_id_num_list.append([temp_leaf_id, chosen_leaf_group_dict[temp_leaf_id]])
        return chosen_leaf_id_num_list

    # 根据要释放的gpu列表释放leaf group资源
    def release_group_with_given_gpu_list(self, released_global_gpu_index_list):
        released_leaf_group_dict = {}
        for temp_global_gpu_index in released_global_gpu_index_list:
            temp_leaf_id = int(temp_global_gpu_index/self.gpu_per_leaf)
            if temp_leaf_id not in released_leaf_group_dict:
                released_leaf_group_dict[temp_leaf_id] = 0
            released_leaf_group_dict[temp_leaf_id] += 1
        # print(released_leaf_group_dict)
        for temp_leaf_id in released_leaf_group_dict:
            self.leaf_list[temp_leaf_id].release_leaf_group_with_required_num(released_leaf_group_dict[temp_leaf_id])

# 测试用
if __name__ == "__main__":
    leaf_resource_manager = LeafResourceManager(4,8)
    print("Step0: ")
    for i in range(4):
        leaf_resource_manager.leaf_list[i].print_resource_info()
    print("Step1: ")
    print(leaf_resource_manager.choose_gpu_in_leaf(16))
    print([leaf_resource_manager.choose_gpu_in_leaf(8),8])
    print([leaf_resource_manager.choose_gpu_in_leaf(8),8])
    print("Step2:")
    leaf_resource_manager.update_group_with_given_gpu_list([16,17,18,19,20,21,22,23])
    leaf_resource_manager.leaf_list[2].print_resource_info()
    leaf_resource_manager.update_group_with_given_gpu_list([24,25,26,27,28,29,30,31])
    leaf_resource_manager.leaf_list[3].print_resource_info()
    print("Step3:")
    leaf_resource_manager.release_group_with_given_gpu_list([i for i in range(32)])
    for i in range(4):
        leaf_resource_manager.leaf_list[i].print_resource_info()
