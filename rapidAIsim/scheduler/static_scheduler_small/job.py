class job:
    def __init__(self,job_id):
        self.id = job_id


        self.allocated_gpus = []
        self.allocated_spine_switches = {}

        self.mini_clos_n = 0
        self.mini_clos_m = 0

        self.min_exception_index = -1
        self.min_exception_num = -1

    # mini clos has the size of n*m
    def set_mini_clos_n(self, n):
        self.mini_clos_n = n

    def add_gpu(self,gpu_index):
        self.allocated_gpus.append(gpu_index)

    def add_spine_switch(self, spine_switch_index, num):
        #if spine_switch_index in self.allocated_spine_switches.keys(): self.allocated_spine_switches[spine_switch_index] += 1
        #else: self.allocated_spine_switches[spine_switch_index] = 1
        #self.allocated_spine_switches.append(spine_switch_index)
        self.allocated_spine_switches[spine_switch_index] = num

    def get_gpu_index(self):
        return self.allocated_gpus