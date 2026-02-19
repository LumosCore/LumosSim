class OXC_Group:
    def __init__(self, id, gpu_num_in_this_group):
        self.id = id
        self.remain_gpu_num_in_this_group = gpu_num_in_this_group
        