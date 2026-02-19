class ELE_Group:
    def __init__(self, id, tor_in_this_group):
        self.id = id
        self.tor_in_this_group = tor_in_this_group
        self.remain_tor_in_this_group = tor_in_this_group
        self.tor_list = [0 for i in range(tor_in_this_group)]
        
    def occupy_tor(self, oxc_id, tor_col_per_oxc, to_occupy_num):
        has_occupy_num = 0
        start_tor_id = oxc_id*tor_col_per_oxc
        end_tor_id = (1+oxc_id)*tor_col_per_oxc
        tor_occupy_list = []
        for tor_id in range(start_tor_id, end_tor_id):
            if has_occupy_num < to_occupy_num and self.tor_list[tor_id] == 0:
                self.tor_list[tor_id] = 1
                has_occupy_num += 1
                tor_occupy_list.append(self.id*self.tor_in_this_group + tor_id)
        assert has_occupy_num == to_occupy_num
        self.remain_tor_in_this_group -= to_occupy_num
        return tor_occupy_list
    
    def release_tor(self, local_tor_id):
        assert self.tor_list[local_tor_id] == 1
        self.remain_tor_in_this_group += 1
        self.tor_list[local_tor_id] = 0