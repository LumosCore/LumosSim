from rapidAIsim.core.simulator import Simulator
from rapidAIsim.core.infrastructure.flow import Flow
import configparser
import random


def test_routing():
    conf_path = 'xxx/rapidAIsim-moe/vclos_exp/ecmp/exp.ini'
    conf_handler = configparser.ConfigParser()
    conf_handler.optionxform = lambda option: option  # Enable case sensitive
    conf_handler.read(conf_path)
    Simulator.setup(conf_handler)
    # Simulator.CONF_DICT['rail_optimized'] = 'yes'
    Simulator.CONF_DICT['find_next_hop_method'] = 'balance'
    Simulator.create_infrastructure()
    Simulator.load_scheduler()
    server_num = Simulator.get_infrastructure().server_num
    for i in range(server_num):
        random.seed(i)
        src = 0
        dst = random.randint(0, 8) + i * 8
        if Simulator.is_in_the_same_server(src, dst):
            continue
        flow = Flow(0, 0, None, src, dst, 0, 0, 0, 0, 8)
        path = flow.find_hop_list()
        print(f"src = {src}, dst = {dst}: {path}")

    # switch_num = Simulator.get_infrastructure().leaf_switch_num
    # for i in range(switch_num * 4):
    #     node = server_num * 8 + i
    #     print(len(Simulator.clos_up_table[node]))
    #     down_table = Simulator.clos_down_table[node]
    #     next_hops = []
    #     for v in down_table.values():
    #         next_hops.extend(v)
    #     print(len(set(next_hops)))
    #     print(len(Simulator.clos_down_table[node]))


if __name__ == '__main__':
    test_routing()
