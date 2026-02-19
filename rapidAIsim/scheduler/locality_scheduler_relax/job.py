class Job:
    def __init__(self,job_id):
        self.id = job_id
        self.start_time = 0.0
        self.finish_time = 0.0
        self.allocated_gpus = []
        self.job_leaf_to_spine_map = {}
        #迁移时可能改变
        self.allocated_oxc_spine_link = {} #allocated_link[oxc_id][leaf_id]=spine_id
        self.used_spine_port_num_pair = {}

    def check_job_allocation_valid(self):
        # for leaf_id in self.job_leaf_to_spine_map:
        #     print(leaf_id,  self.job_leaf_to_spine_map[leaf_id])
        leaf_num_map = {}
        spine_num_map = {}
        for leaf_id in self.job_leaf_to_spine_map:
            for spine_id in  self.job_leaf_to_spine_map[leaf_id]:
                if(self.job_leaf_to_spine_map[leaf_id][spine_id]>0):
                    if leaf_id not in leaf_num_map:
                        leaf_num_map[leaf_id] = 0
                    leaf_num_map[leaf_id]+=self.job_leaf_to_spine_map[leaf_id][spine_id]
                    if spine_id not in spine_num_map:
                        spine_num_map[spine_id] = 0
                    spine_num_map[spine_id]+=self.job_leaf_to_spine_map[leaf_id][spine_id]
        for item in leaf_num_map:
            assert leaf_num_map[item]<=64
        for item in spine_num_map:
            assert spine_num_map[item]<=64

        spine_portNum_map = {}
        for oxc_id in self.allocated_oxc_spine_link:
            for leaf_id in self.allocated_oxc_spine_link[oxc_id]:
                spine_id = self.allocated_oxc_spine_link[oxc_id][leaf_id]
                if spine_id not in spine_portNum_map:
                    spine_portNum_map[spine_id] = 0
                spine_portNum_map[spine_id] += 1

        # print(spine_portNum_map)
        # print(spine_num_map)
        temp_size = 0
        for item in spine_num_map:
            assert spine_portNum_map[item]==spine_num_map[item]
            assert spine_portNum_map[item]==self.used_spine_port_num_pair[item]
            temp_size+=spine_portNum_map[item]
        print(self.id, temp_size, len(self.allocated_gpus))
        assert(temp_size==0 or temp_size==len(self.allocated_gpus))

