

class HwOxcAll2allScheduler:
    def __init__(self, leaf_switch_num, leaf_switch_port_num, downlinks, NIC_num) -> None:
        self._leaf_switch_num = leaf_switch_num
        self._leaf_switch_port_num = leaf_switch_port_num
        self._downlinks = downlinks
        self._NIC_num = NIC_num

        self._record_occupied_NIC_set = set()
        self._task_NIC_map = dict()    # Record NIC id used by every task.
        self._virtual_switch_leisure_NIC_num_map = dict()    # Record the number of unoccupied NIC belonging to every switch.
        for i in range(NIC_num, NIC_num + leaf_switch_num):
            self._virtual_switch_leisure_NIC_num_map[i] = self._downlinks


    def schedule(self, task_occupied_NIC_num, taskid, current_time, waiting_task_list):
        from rapidAIsim.core.simulator import Simulator
        allocate_succeed = False
        need_NIC_list = None
        allocated_link_mapping = None
        all_gpu_index = None
        link_mapping = None
        downlinks = self._downlinks

        unoccpuied_NIC_set = self.get_leisure_NIC_set()

        if task_occupied_NIC_num > len(unoccpuied_NIC_set):
            allocate_succeed = False
        else:
            need_NIC_list = None
            need_free_switch_num = int(task_occupied_NIC_num / downlinks)
            tmp_NIC_record = []
            residual_need_num = task_occupied_NIC_num
            if task_occupied_NIC_num >= downlinks:
                #  Preferentially assign NICs to the full-free switch.
                for switch_id, leisure_num in self._virtual_switch_leisure_NIC_num_map.items():
                    if leisure_num == downlinks:
                        tmp_NIC_record += self.get_NIC_list_in_switch(switch_id, downlinks, self._NIC_num, downlinks)
                        self._virtual_switch_leisure_NIC_num_map[switch_id] -= downlinks
                        need_free_switch_num -= 1
                        residual_need_num -= downlinks
                    if need_free_switch_num == 0:
                        break
            # task_occupied_NIC_num < downlinks
            else:
                # Reversely sort by NIC_num.
                switch_leisure_NIC_num_map_list = sorted(self._virtual_switch_leisure_NIC_num_map.items(), key = lambda d:d[1], reverse = True)
                # Preferentially assign NICs to the one switch.
                # Meanwhile do not waste full-free switches.

                if residual_need_num > 0:
                    for (switch_id, leisure_num) in switch_leisure_NIC_num_map_list:
                        if leisure_num < downlinks and leisure_num > residual_need_num:
                            tmp_NIC_record += self.get_NIC_list_in_switch(switch_id, downlinks, self._NIC_num, residual_need_num)
                            self._virtual_switch_leisure_NIC_num_map[switch_id] -= residual_need_num
                            residual_need_num = 0
                            break

                    if residual_need_num > 0:
                        for (switch_id, leisure_num) in switch_leisure_NIC_num_map_list:
                            if leisure_num == downlinks:
                                self._virtual_switch_leisure_NIC_num_map[switch_id] -= residual_need_num
                                tmp_NIC_record += self.get_NIC_list_in_switch(switch_id, downlinks, self._NIC_num, residual_need_num)
                                residual_need_num = 0
                                break
                    
                    if residual_need_num > 0:
                        for (switch_id, leisure_num) in switch_leisure_NIC_num_map_list:
                            self._virtual_switch_leisure_NIC_num_map[switch_id] -= leisure_num
                            tmp_NIC_record += self.get_NIC_list_in_switch(switch_id, downlinks, self._NIC_num, leisure_num)
                            residual_need_num -= leisure_num

            
            if residual_need_num == 0:
                need_NIC_list = tmp_NIC_record
            else:
                # Cannot allocate a new task. Rollback!
                for NIC_id in tmp_NIC_record:
                    self._virtual_switch_leisure_NIC_num_map[self.belong_which_leaf_switch(NIC_id)] += 1
                allocate_succeed = False
                return allocate_succeed, need_NIC_list, allocated_link_mapping, all_gpu_index, link_mapping


            self._task_NIC_map[taskid] = need_NIC_list
            allocated_link_mapping = []

            leaf_switch_list = []
            switch_NIC_cnt = {}
            for NIC_id in need_NIC_list:
                switch_id = self.belong_which_leaf_switch(NIC_id)
                if switch_NIC_cnt.get(switch_id):
                    switch_NIC_cnt[switch_id] += 1
                else:
                    switch_NIC_cnt[switch_id] = 1

                leaf_switch_list.append(switch_id)
                # Links between NICs and leaf switches.
                # allocated_link_mapping.append((NIC_id, switch_id, 2))
                # allocated_link_mapping.append((switch_id, NIC_id, 2))

                # self._record_occupied_NIC_set.add(NIC_id)


            leaf_switch_set = set(leaf_switch_list)
            leaf_switch_list = sorted(list(leaf_switch_set))
            switch_num = len(leaf_switch_list)
            
            if switch_num == 2:
                pass
                # allocated_link_mapping.append((leaf_switch_list[0], leaf_switch_list[1], downlinks))
                # allocated_link_mapping.append((leaf_switch_list[1], leaf_switch_list[0], downlinks))
            elif switch_num > 2:
                for i in range(switch_num):
                    src = leaf_switch_list[i]
                    if i == switch_num - 1:
                        dst = leaf_switch_list[0]
                    else:
                        dst = leaf_switch_list[i + 1]
                    # allocated_link_mapping.append((src, dst, downlinks))
                    # allocated_link_mapping.append((dst, src, downlinks))
            allocate_succeed = True
            Simulator.TASK_SWITCH_DICT[taskid] = leaf_switch_list
            Simulator.TASK_NIC_DICT[taskid] = need_NIC_list

            # ---- Actually effect allocated_link_mapping -----
            # Detect the number of occupied servers.
            NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
            need_server_num = int(task_occupied_NIC_num / NIC_num_in_a_server)
            # Regular linking
            for src in need_NIC_list:
                src_belong = src // NIC_num_in_a_server
                src_port_serial = src % NIC_num_in_a_server
                dst = NIC_num_in_a_server * src_port_serial + src_belong
                allocated_link_mapping.append((src, dst, 1))
                self._record_occupied_NIC_set.add(src)
        return allocate_succeed, need_NIC_list, allocated_link_mapping, all_gpu_index, link_mapping


    def update_finished_job(self, taskid, current_time, waiting_task_list):
        NIC_list = self._task_NIC_map[taskid]
        for NIC_id in NIC_list:
            self._record_occupied_NIC_set.remove(NIC_id)
            self._virtual_switch_leisure_NIC_num_map[self.belong_which_leaf_switch(NIC_id)] += 1


    def belong_which_leaf_switch(self, NIC_id):
        return NIC_id // self._downlinks + self._NIC_num


    def get_leisure_NIC_set(self):
        all_NIC_set = set([i for i in range(self._NIC_num)])
        return all_NIC_set ^ self._record_occupied_NIC_set


    def get_NIC_list_in_switch(self, switch_id, downlinks, NIC_num, need_num):
        if need_num == downlinks:
            return [i for i in range((switch_id - NIC_num) * downlinks, (switch_id - NIC_num + 1) * downlinks)]
        elif need_num < downlinks:
            leisure_NIC_set = self.get_leisure_NIC_set()
            switch_NIC_set = set([i for i in range((switch_id - NIC_num) * downlinks, (switch_id - NIC_num + 1) * downlinks)])
            
            can_be_used = switch_NIC_set & leisure_NIC_set
            res = []
            for NIC_id in can_be_used:
                res.append(NIC_id)
                need_num -= 1
                if need_num <= 0:
                    break
            return res
        elif need_num > downlinks:
            print("Bug: get_NIC_list_in_switch need_num > downlinks")
            return