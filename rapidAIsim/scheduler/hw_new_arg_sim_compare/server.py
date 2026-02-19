class Server:
    def __init__(self, id, gpu_per_server, belong_lead_id, belong_spine_id):
        self.remain_gpu_num = gpu_per_server
        self.server_id = id
        self.belong_lead_id = belong_lead_id
        self.belong_spine_id = belong_spine_id
