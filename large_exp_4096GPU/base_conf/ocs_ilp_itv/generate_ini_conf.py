import sys
from jinja2 import Environment, PackageLoader
from base_conf_template.create_dragonfly import create_pod_connect_list
from base_conf_template.generate_tasks import get_fixed_requests_256_part_llm_tasks

if __name__ == '__main__':
    try:
        beta = int(sys.argv[1])
    except IndexError:
        beta = 3000
    NIC_num = 4096
    leaf_switch_num = 256
    leaf_switch_port_num = 32
    spine_switch_num = 256
    spine_switch_port_num = 32
    leaf_spine_link_num = 2
    NIC_num_in_a_server = 8
    ocs_num = 128
    pod_num = 32
    server_num_per_pod = 16

    joint_scheduler = 'OCSExpander'


    task_list = get_fixed_requests_256_part_llm_tasks(1000, beta)


    if joint_scheduler == 'static':
        connect_info_list = create_pod_connect_list()
        connect_info_str = str(connect_info_list)
    else:
        connect_info_str = []

    env = Environment(loader=PackageLoader('base_conf_template', './'))
    template = env.get_template('base_ini_template.j2')
    content = template.render(
        ocs_num=ocs_num,
        pod_num=pod_num,
        layers=3,
        ocs_reconfiguration='yes',
        server_num_per_pod=server_num_per_pod,
        connect_info_str=connect_info_str,
        topo_type='clos',
        find_path_method='updown',  
        joint_scheduler=joint_scheduler,
        measure_sampling_interval=10,
        greedy='yes',
        find_next_hop_method='ecmp',  
        max_rehashing_time=1,
        waiting_task_order_mode='FIFO',
        NIC_num=NIC_num,
        NIC_num_in_a_server=NIC_num_in_a_server,
        leaf_switch_num=leaf_switch_num,
        leaf_switch_port_num=leaf_switch_port_num,
        spine_switch_port_num=spine_switch_port_num,
        spine_switch_num=spine_switch_num,
        leaf_spine_link_num=leaf_spine_link_num,
        inner_server_bandwidth=2000,
        switch_port_bandwidth=200,
        computation_time=1,
        task_type='llm',
        task_list=task_list,
        task_iteration_num=10,
        reconfiguration='no',
        rail_optimized='yes',
        strategy='ilp_itv',
        need_comm_orc='no',
    )
    with open('exp.ini', 'w') as f:
        f.write(content)
