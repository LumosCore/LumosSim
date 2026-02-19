import random
from rapidAIsim.communication_strategy.strategy_base import StrategyBase
from rapidAIsim.core.infrastructure.flow import Flow

VCLOS_MODEL_LIST = ['Bert', 'ResNet50', 'ResNet101', 'VGG16', 'Pangu', 'llama_7b', 'llama2_7b', 'llama2_13b', 'GPT2_13b', 'deepseek3', 'gpt4', 'llama2_70b', 'deepseek_tuili']
ALL_REDUCE_COST = {"Pangu": 0.0737, "llama_7b": 0.189, "llama2_7b": 0.111, "llama2_13b": 0.0087, "GPT2_13b": 0.047, "deepseek3": 0.0737, "gpt4": 0.047,"llama2_70b": 0.0087, "deepseek_tuili": 0.00001,}
ALL2ALL_COST = {"Pangu": 0.258, "llama_7b": 0.0, "llama2_7b": 0.0, "llama2_13b": 0.0, "GPT2_13b": 0.195, "deepseek3": 0.258, "gpt4": 0.195, "llama2_70b": 0.0, "deepseek_tuili": 0.0737,}
PP_COST = {"Pangu": 0.0058, "llama_7b": 0.0077, "llama2_7b": 0.0047, "llama2_13b": 0.0034, "GPT2_13b": 0.0063, "deepseek3": 0.0058, "gpt4": 0.0063, "llama2_70b": 0.0034, "deepseek_tuili": 0.00001,}


def get_vclos_model_info():
    running_map = {}
    running_with_contention_map = {}

    running_map[('llama_7b',16)] = 520
    running_map[('llama2_7b',16)] = 1074
    running_map[('llama2_13b',16)] = 1689
    running_map[('Pangu',16)] = 278

    running_map[('llama_7b',32)] = 520
    running_map[('llama2_7b',32)] = 1074
    running_map[('llama2_13b',32)] = 1689
    running_map[('Pangu',32)] = 278

    running_map[('llama_7b',64)] = 575
    running_map[('llama2_7b',64)] = 1082
    running_map[('llama2_13b',64)] = 1717
    running_map[('GPT2_13b',64)] = 903
    running_map[('Pangu',64)] = 279

    running_map[('llama_7b',96)] = 602.13
    running_map[('llama2_7b',96)] = 1103.38
    running_map[('llama2_13b',96)] = 1763.47
    running_map[('GPT2_13b',96)] = 956.83
    running_map[('Pangu',96)] = 371.78

    running_map[('llama_7b',128)] = 611
    running_map[('llama2_7b',128)] = 1110
    running_map[('llama2_13b',128)] = 1782
    running_map[('GPT2_13b',128)] = 963
    running_map[('Pangu',128)] = 424

    running_map[('VGG16',128)] = 153.531218
    running_map[('VGG16',96)] = 152.1298174
    running_map[('VGG16',64)] = 150.7537688
    running_map[('VGG16',32)] = 149.775337
    running_map[('VGG16',16)] = 146.8428781

    running_map[('ResNet50',128)] = 113.7656428
    running_map[('ResNet50',96)] = 112.7395716
    running_map[('ResNet50',64)] = 111.6902457
    running_map[('ResNet50',32)] = 111.2347052
    running_map[('ResNet50',16)] = 108.5383502

    running_map[('ResNet101',128)] = 195.9503592
    running_map[('ResNet101',96)] = 196.3350785
    running_map[('ResNet101',64)] = 196.0784314
    running_map[('ResNet101',32)] = 189.3939394
    running_map[('ResNet101',16)] = 187.1490954

    running_map[('Bert',128)] = 538.2131324
    running_map[('Bert',96)] = 531.3496281
    running_map[('Bert',64)] = 524.2005941
    running_map[('Bert',32)] = 514.7563487
    running_map[('Bert',16)] = 492.0452682

    running_with_contention_map[('llama_7b',32)] = 643
    running_with_contention_map[('llama2_7b',32)] = 1173
    running_with_contention_map[('llama2_13b',32)] = 1815
    running_with_contention_map[('Pangu',32)] = 328

    running_with_contention_map[('llama_7b',96)] = 791.62
    running_with_contention_map[('llama2_7b',96)] = 1163.24
    running_with_contention_map[('llama2_13b',96)] = 1886.59
    running_with_contention_map[('GPT2_13b',96)] = 1160.3
    running_with_contention_map[('Pangu',96)] = 488.74

    running_with_contention_map[('VGG16',128)] = 167.5041876
    running_with_contention_map[('VGG16',96)] = 165.7458564
    running_with_contention_map[('VGG16',64)] = 162.7780792
    running_with_contention_map[('VGG16',32)] = 158.4786054
    running_with_contention_map[('VGG16',16)] = 141.1100659

    running_with_contention_map[('ResNet50',128)] = 114.6350783
    running_with_contention_map[('ResNet50',96)] = 114.2421935
    running_with_contention_map[('ResNet50',64)] = 114.1552511
    running_with_contention_map[('ResNet50',32)] = 113.4215501
    running_with_contention_map[('ResNet50',16)] = 109.7694841

    running_with_contention_map[('ResNet101',128)] = 195.4397394
    running_with_contention_map[('ResNet101',96)] = 196.4636542
    running_with_contention_map[('ResNet101',64)] = 197.3684211
    running_with_contention_map[('ResNet101',32)] = 194.0491591
    running_with_contention_map[('ResNet101',16)] = 189.3939394

    running_with_contention_map[('Bert',128)] = 437.4453193
    running_with_contention_map[('Bert',96)] = 433.9022274
    running_with_contention_map[('Bert',64)] = 430.9106579
    running_with_contention_map[('Bert',32)] = 423.9084358
    running_with_contention_map[('Bert',16)] = 411.8616145

    comm_all_ration_map = {}
    for model_name in VCLOS_MODEL_LIST:
        for beta in [8, 16, 32, 64, 96, 128]:
            if (model_name, beta) in running_map:
                beta = max(8,beta)
                y_1 = running_with_contention_map[(model_name, beta)]
                y_0 = running_map[(model_name, beta)]
                comm_all_ration_map[(model_name, beta)] = (y_1 - y_0) / y_0
 
    all_reduce_cost = {}
    for (model_name, beta), ratio in comm_all_ration_map:
        all_reduce_cost[(model_name, beta)] = ratio * ALL_REDUCE_COST['Pangu'] / (ALL_REDUCE_COST['Pangu'] + ALL2ALL_COST['Pangu'])
    return all_reduce_cost


class LLM(StrategyBase):
    def __init__(self):
        super().__init__()
        self._dp_pp_rounds = []
        self._actual_round_pair_list = []
        self._round_pair_list = []
        self.model_type = ''
        self.TP = 0
        self.DP = 0
        self.PP = 0
        self.EP = 0
        self.total_computation_time = 0

    def deal_job(self, task_id, model_size, task_occupied_NIC_num, use_NIC_list=None, NIC_num_in_a_server=8, **kwargs):
        """
        The LLM strategy. Kwargs should contain `ep_qp`, `pp_qp`, `dp_qp`, or raise a TypeError.
        LLM strategy imitates an iteration of LLM training, including forward and backward propagation.
        In forward propagation, PP traffic and EP traffic are transmitted sequentially. Currently, MOE happens in the
        second round of PP traffic, if PP >= 2. In backward propagation, DP and PP traffic is transmitted
        simultaneously, while different DP domains are transmitted sequentially.

        The total process is like (if PP=4, DP=4 and the model has EP communication):
        PP -> EP -> PP -> PP -> DP+PP -> DP+PP -> EP -> DP+PP -> DP. Computation is also considered, but cannot adapt to
        different communication patterns. The communication is divided into several rounds, the flows in a round starts
        at the same time, and one computation happens before every round start (implemented in
        `stage_controller.del_global_record_trigger_new_round`).

        The DP traffic only contains one stage in each round, because the communication is the same in different stages.
        Therefore, the DP traffic size is multiplied by the number of stages in a round, that is, the size of DP - 1.
        """
        try:
            ep_qp, pp_qp, dp_qp = kwargs['ep_qp'], kwargs['pp_qp'], kwargs['dp_qp']
        except KeyError:
            raise TypeError('ep_qp, pp_qp, dp_qp are required in arguments')

        from rapidAIsim.core.simulator import Simulator
        from rapidAIsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {task_id}')
        Simulator.task_time_logger.write(f'taskid,{task_id},start_time,{Simulator.get_current_time()}\n')

        conservative = Simulator.CONF_DICT['find_next_hop_method'] == 'conservative'
        task = Simulator.TASK_LIST[task_id]
        model_type = task.model_type
        duration_time = task.duration_time
        iteration_num = task.task_iteration_num
        one_iter_duration_time = duration_time / iteration_num
        bandwidth = float(Simulator.CONF_DICT['switch_port_bandwidth'])
        TP, PP, DP, EP = task.TP, task.PP, task.DP, task.EP
        self.model_type = model_type
        self.TP, self.PP, self.DP, self.EP = TP, PP, DP, EP
        pp_size, dp_size, ep_size, one_iter_total_computation_time = self.calculate_comm_size_and_comp_time(
            TP, PP, DP, EP, model_type, one_iter_duration_time)
        if task_id in Simulator.SCHEDULER_TIME_COST:
            one_iter_total_computation_time += Simulator.SCHEDULER_TIME_COST[task_id] / iteration_num
        print(f"debug: LLM strategy, dp_size={DP}, pp_size={PP}, ep_size={EP}\n")
            #   f"total_computation_time={total_computation_time}, "
            #   f"duration_time={duration_time}, computaion_ratio={total_computation_time/duration_time}")
        # print("debug pp_qp")
        # print(pp_qp)
        # print("debug dp_qp")
        # print(dp_qp)
        # print("debug ep_qp")
        # print(ep_qp)

        round_pair_list = self.get_original_pair_list(task_id, pp_qp, dp_qp, ep_qp, pp_size, dp_size, ep_size)
        flow_num = Simulator.FLOWID
        # 生成flow，并加入等待传输流队列
        infra = Simulator.get_infrastructure()
        gpu_num_per_leaf = infra.NIC_num // infra.leaf_switch_num
        actual_round_pair_list = []
        actual_round_id = 0
        computation_round = 0  # 统计计算的轮数。
        for round_id, round_pair in enumerate(round_pair_list):
            waited_flows = []
            inter_leaf_flows = []
            intra_leaf_flows = []
            for src, dst, size, prior_calculate, subsequent_calculate in round_pair:
                if src // NIC_num_in_a_server == dst // NIC_num_in_a_server:
                    continue
                if src // gpu_num_per_leaf == dst // gpu_num_per_leaf:
                    intra_leaf_flows.append([src, dst, size, prior_calculate, subsequent_calculate])
                else:
                    inter_leaf_flows.append([src, dst, size, prior_calculate, subsequent_calculate])
            if not inter_leaf_flows and not intra_leaf_flows:
                continue
            while True:
                if not inter_leaf_flows:
                    # 如果没有跨leaf的流，直接添加一个intra_leaf_flow
                    waited_flows.append(intra_leaf_flows[0])
                    break
                if intra_leaf_flows and round_id in self._dp_pp_rounds and DP > 1:
                    # 对于PP+DP round，如果删除所有的intra_leaf_flow，会导致DP和PP的冲突无法反映在最终的结果中。
                    # 因此这种情况需要在添加所有的inter_leaf_flows的同时，添加一个intra_leaf_flow
                    waited_flows.extend(inter_leaf_flows)
                    intra_leaf_flow_not_found = True
                    i = 0
                    while intra_leaf_flow_not_found and i < len(intra_leaf_flows):
                        src_gpu = intra_leaf_flows[i][0]
                        i += 1
                        for flow_info in intra_leaf_flows:
                            if flow_info[0] == src_gpu:
                                waited_flows.append(flow_info)
                                intra_leaf_flow_not_found = False
                                break
                else:
                    waited_flows.extend(inter_leaf_flows)
                break

            actual_round_pair_list.append([])
            round_prior = False
            round_subsequent = False
            for src, dst, size, prior, subsequent in waited_flows:
                round_prior = round_prior or prior
                round_subsequent = round_subsequent or subsequent
                actual_round_pair_list[-1].append([src, dst, size, prior, subsequent])
                flow = Flow(Simulator.FLOWID, size, None, int(src), int(dst), size, None, task_id, actual_round_id,
                            task_occupied_NIC_num, conservative, need_prior_calculate=prior,
                            need_subsequent_calculate=subsequent)
                self.record_network_occupy(task_id, actual_round_id, flow, src)
                Simulator.task_need_comm_size[task_id] += size * iteration_num
                Simulator.FLOWID += 1
            if waited_flows:
                actual_round_id += 1
            computation_round += int(round_prior) + int(round_subsequent)
           
            # assert actual_round_id == round_id + 1
        flow_num = Simulator.FLOWID - flow_num
        Simulator.task_expected_comm_time[task_id] = Simulator.task_need_comm_size[task_id] / bandwidth

        # 均分每一个computation的时间
        computation_time = one_iter_total_computation_time / computation_round
        task.computation_time = computation_time
        task.computation_round = computation_round
        Simulator.task_need_comp_time[task_id] = one_iter_total_computation_time * iteration_num

        if Simulator.get_wait_transmit_dict().get(f'{task_id}_0') is None:
            Simulator.get_wait_transmit_dict()[f'{task_id}_0'] = {}

        # Register first round job flows
        flow_list = list(Simulator.get_wait_transmit_dict()[f'{task_id}_0'].values())
        # print(f'debug round 0 flow {flow_list[0].get_flow_id()} of task {taskid} with computation {computation_time}')
        sync_delay_alpha = 0
        if Simulator.CONF_DICT['joint_scheduler'] in ['OCSExpander_superPod','OCSExpander']:
            sync_delay_alpha = 3*float(Simulator.CONF_DICT['sync_delay_alpha'])
        elif Simulator.CONF_DICT['joint_scheduler'] in ['ELEExpander','ELEExpander_SuperPod']:
            sync_delay_alpha = 4*float(Simulator.CONF_DICT['sync_delay_alpha'])
        # for flow in flow_list:
        #     print("debug sync_delay_alpha",sync_delay_alpha, computation_time,flow._flow_id, flow._src,  flow._dst, flow._size)
        Simulator.register_event(FlowTransmitEvent(computation_time + sync_delay_alpha, flow_list))
        self._actual_round_pair_list = actual_round_pair_list

        # print debug info
        try:
            ep_flow_shape = len(ep_qp), len(ep_qp[0])
        except IndexError:
            ep_flow_shape = 0, 0
        try:
            pp_flow_shape = len(pp_qp), len(pp_qp[0])
        except IndexError:
            pp_flow_shape = 0, 0
        try:
            dp_flow_shape = len(dp_qp), len(dp_qp[0])
        except IndexError:
            dp_flow_shape = 0, 0
        print(f"Task {task_id} info: TP={TP}, PP={PP}, DP={DP}, EP={EP}")
        print(f"Task {task_id} QP info: ep_flow_shape: {ep_flow_shape}, pp_flow_shape: {pp_flow_shape},"
              f" dp_flow_shape: {dp_flow_shape}")
        # dp, pp, ep = self.get_one_iteration_flow_num(TP, PP, DP, EP)
        # print(f"Task {taskid} flow num (calculated): dp={dp}, pp={pp}, ep={ep}, total={dp + pp + ep}")
        print(f"Task {task_id} flow num (actual): {flow_num}")

    def get_original_pair_list(self, task_id, pp_qp, dp_qp, ep_qp, pp_size, dp_size, ep_size):
        """
        根据调度器生成的queue pair，计算每一种通信的数据量，并生成每一轮的通信对列表。
        当前的设计：只有反向传播的EP，且仅挑选一个round的EP进行实际的流生成。
        如果更改了EP的设计，请修改当前的文档。
        """
        print("debug get_original_pair_list")
        TP, PP, DP, EP = self.TP, self.PP, self.DP, self.EP

        round_pair_list = []
        # Forward: PP -> EP -> PP -> PP ...
        round_id = 0
        # 第一个PP
        if PP >= 2:
            round_pair_list.append([])
            for src, dst in pp_qp[0]:
                round_pair_list[-1].append((src, dst, pp_size, True, False))
            # print(f"round_id={round_id}, flow_nums={len(round_pair_list[-1])}")
            round_id += 1

        # 剩余的PP
        for i in range(1, PP - 1):
            round_pair_list.append([])
            if i == PP - 2:
                for src, dst in pp_qp[i]:
                    round_pair_list[-1].append((src, dst, pp_size, True, True))
            else:
                for src, dst in pp_qp[i]:
                    round_pair_list[-1].append((src, dst, pp_size, True, False))
            # print(f"round_id={round_id}, flow_nums={len(round_pair_list[-1])}")
            round_id += 1
        # Backward: DP+PP -> DP+PP -> EP -> DP+PP -> DP...
        dp_qp.reverse()
        dp_pp_rounds = []
        # 做DP+PP直到倒数第二层的EP前
        for i in range(PP - 2):
            round_pair_list.append([])
            if dp_qp:
                for src, dst in dp_qp[i]:
                    round_pair_list[-1].append((src, dst, dp_size, True, False))
            for src, dst in pp_qp[PP + i]:
                round_pair_list[-1].append((src, dst, pp_size, True, False))
            # print(f"round_id={round_id}, flow_nums={len(round_pair_list[-1])}")
            dp_pp_rounds.append(round_id)
            round_id += 1
        # 一次EP

        print("debug ep_qp")
        from rapidAIsim.core.simulator import Simulator
        gpu_num_per_pod = int(Simulator.CONF_DICT['server_num_per_pod'])*int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        if len(ep_qp) != 0:
            ep_round_pairs = []
            for i in range(len(ep_qp)):
                ep_round_pairs.append([])
                if i == 0:
                    for src, dst in ep_qp[i]:
                        ep_round_pairs[-1].append((src, dst, ep_size, True, False))
                else:
                    for src, dst in ep_qp[i]:
                        ep_round_pairs[-1].append((src, dst, ep_size, False, False))
                print(i,src, dst, ep_size, src//gpu_num_per_pod, dst//gpu_num_per_pod)
            print("debug ep_round_pairs",len(ep_round_pairs))
            # print(f"round_id={round_id}, flow_nums={len(round_pair_list[-1])}")
            random.seed(task_id)
            round_pair_list.append([])
            round_pair_list[-1].extend(random.choice(ep_round_pairs))
            round_num = len(ep_round_pairs)
            for i in range(len(round_pair_list[-1])):
                src, dst, ep_size, prior_calculate, subsequent_calculate = round_pair_list[-1][i]
                round_pair_list[-1][i] = (src, dst, ep_size * round_num * 2, prior_calculate, subsequent_calculate)
            round_id += 1
        # 剩余的DP+PP（如果PP>=2才会有，同时也是最后一轮PP）
        if PP >= 2:
            round_pair_list.append([])
            # DP
            if dp_qp:
                for src, dst in dp_qp[PP - 2]:
                    round_pair_list[-1].append((src, dst, dp_size, True, False))
            # PP
            for src, dst in pp_qp[-1]:
                round_pair_list[-1].append((src, dst, pp_size, True, False))
            # print(f"round_id={round_id}, flow_nums={len(round_pair_list[-1])}")
            dp_pp_rounds.append(round_id)
            round_id += 1
        # 最后一轮DP
        if dp_qp:
            round_pair_list.append([])
            for src, dst in dp_qp[-1]:
                round_pair_list[-1].append((src, dst, dp_size, True, False))
            # print(f"round_id={round_id}, flow_nums={len(round_pair_list[-1])}")
            round_id += 1
        self._round_pair_list = round_pair_list
        self._dp_pp_rounds = dp_pp_rounds
        return round_pair_list

    def get_task_a_iteration_pair_list(self, *args, **kwargs):
        return self._actual_round_pair_list

    @staticmethod
    def get_one_iteration_flow_num(TP, PP, DP, EP):
        if DP == 1:
            dp = 0
        else:
            dp = DP * PP * TP
        pp = (PP - 1) * TP * DP * 2
        # ep = EP * (EP - 1) * (PP * TP * DP // EP)
        if EP == PP * DP:
            ep = 0
        else:
            ep_gpu_nums = DP * PP * TP // EP
            ep = ep_gpu_nums * (ep_gpu_nums - 1) * EP * 2
        return dp, pp, ep

    @staticmethod
    def calculate_comm_size_and_comp_time(TP, PP, DP, EP, model_type, duration_time):
        """
        计算单个iteration的通信量和计算量。duration_time是一个iteration的时间，单位为秒。
        """
        # 计算PP的通信量
        from rapidAIsim.core.simulator import Simulator
        if  Simulator.CONF_DICT['only_comm'] == 'yes':
            pp_ratio = PP_COST[model_type] *3
        elif  Simulator.CONF_DICT['super_pod'] == 'yes':
            pp_ratio = PP_COST[model_type] /8 
        else:
            pp_ratio = PP_COST[model_type]
        if PP > 1:
            pp_round = 2 * (PP - 1)
            pp_size = duration_time * pp_ratio * 200 / pp_round
        else:
            pp_size = 0
            pp_ratio = 0
        # 计算DP的通信量
        if  Simulator.CONF_DICT['only_comm'] == 'yes':
            dp_ratio = ALL_REDUCE_COST[model_type] *3
        elif  Simulator.CONF_DICT['super_pod'] == 'yes':
            dp_ratio = ALL_REDUCE_COST[model_type] /8 
        else:
            dp_ratio = ALL_REDUCE_COST[model_type]
        if DP > 1:
            dp_size = duration_time * dp_ratio * 200 / (TP * (DP - 1))
        else:
            dp_size = 0
            dp_ratio = 0
        if Simulator.CONF_DICT['rail_optimized'] == 'yes':
            dp_size /= 2
        # 计算EP的通信量
        if  Simulator.CONF_DICT['only_comm'] == 'yes':
            ep_ratio = ALL2ALL_COST[model_type] *3
        elif  Simulator.CONF_DICT['super_pod'] == 'yes':
            ep_ratio = ALL2ALL_COST[model_type] /8 
        else:
            ep_ratio = ALL2ALL_COST[model_type]
        if EP == TP:
            ep_size = 0
            ep_ratio = 0
        else:
            ep_size = duration_time * ep_ratio * 200 / (EP - 1)
            
        if  Simulator.CONF_DICT['only_comm'] == 'yes':
            remain_ratio = 1 - pp_ratio - ep_ratio - dp_ratio
            dp_ratio += remain_ratio/3
            ep_ratio += remain_ratio/3
            pp_ratio += remain_ratio/3
        computation_raio = max(0.000001,1 - pp_ratio - dp_ratio - ep_ratio)
        total_computation_time = duration_time * computation_raio
        return pp_size, dp_size, ep_size, total_computation_time
