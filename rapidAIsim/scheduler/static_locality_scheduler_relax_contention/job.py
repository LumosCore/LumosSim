class Job:
    def __init__(self,job_id):
        self.id = job_id
        self.start_time = 0.0
        self.finish_time = 0.0
        self.allocated_gpus = []
        #迁移时可能改变
        self.job_allocated_leaf_spine_link = {} #allocated_link[oxc_id][leaf_id]=spine_id

