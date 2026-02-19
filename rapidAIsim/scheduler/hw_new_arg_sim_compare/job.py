class Job:
    def __init__(self,job_id, used_tor_num):
        self.id = job_id
        self.start_time = 0.0
        self.finish_time = 0.0
        self.used_tor_num = used_tor_num
        self.allocated_tors = []
        