

def create_clos_connect_list(NIC_num, leaf_switch_num, spine_switch_num, leaf_switch_port_num, spine_switch_port_num, NIC_num_in_a_server):
    """
    The ID of NIC starts from 0.
    The ID of switch starts from the maximum NIC ID + 1.

    Return:
    - connect_info_list: [(src_id, dst_id, link_num)]
    """

    assert (NIC_num == int(leaf_switch_port_num / 2) * leaf_switch_num)
    assert (spine_switch_port_num >= leaf_switch_num)

    connect_info_list = []

    # {key(switch id): value(global device id)} 
    leaf_switch_dict = {}
    spine_switch_dict = {}
    for i in range(leaf_switch_num):
        leaf_switch_dict[i] = i + NIC_num

    for i in range(spine_switch_num):
        spine_switch_dict[i] = i + NIC_num + leaf_switch_num
    
    # Add links between NICs and leaf switches.
    leaf_switch_id = 0
    for connected_NIC_id in range(NIC_num):
        connect_info_list.append((connected_NIC_id, leaf_switch_dict[leaf_switch_id], 1))
        connect_info_list.append((leaf_switch_dict[leaf_switch_id], connected_NIC_id, 1))
        if connected_NIC_id > 0 and (connected_NIC_id + 1) % int(leaf_switch_port_num / 2) == 0:
            leaf_switch_id += 1

    assert leaf_switch_id == leaf_switch_num
    
    # Add links between spine switches and leaf switches.
    for leaf_switch_id in range(leaf_switch_num):
        for spine_switch_id in range(spine_switch_num):
            connect_info_list.append((spine_switch_dict[spine_switch_id], leaf_switch_dict[leaf_switch_id], int(leaf_switch_port_num / 2 / spine_switch_num)))
            connect_info_list.append((leaf_switch_dict[leaf_switch_id], spine_switch_dict[spine_switch_id], int(leaf_switch_port_num / 2 / spine_switch_num)))

    print('connection num:', len(connect_info_list))
    return connect_info_list


def create_hw_eps_connect_list(NIC_num, leaf_switch_num, spine_switch_num, leaf_switch_port_num, spine_switch_port_num, NIC_num_in_a_server):
    """
    isolated 2 plane

    The ID of NIC starts from 0.
    The ID of switch starts from the maximum NIC ID + 1.

    Return:
    - connect_info_list: [(src_id, dst_id, link_num)]
    """

    connect_info_list = []
    downlink_num = int(leaf_switch_port_num / 2)

    # {key(switch id): value(global device id)} 
    leaf_switch_dict = {}
    spine_switch_dict = {}
    for i in range(leaf_switch_num):
        leaf_switch_dict[i] = i + NIC_num

    for i in range(spine_switch_num):
        spine_switch_dict[i] = i + NIC_num + leaf_switch_num

    # Add links between NICs and leaf switches.
    leaf_switch_id = 0
    multiple_link_num = int(NIC_num_in_a_server / downlink_num)
    for connected_NIC_id in range(NIC_num):
        connect_info_list.append((connected_NIC_id, leaf_switch_dict[leaf_switch_id], multiple_link_num))
        connect_info_list.append((leaf_switch_dict[leaf_switch_id], connected_NIC_id, multiple_link_num))
        if connected_NIC_id > 0 and (connected_NIC_id + 1) % int(leaf_switch_port_num / 2) == 0:
            leaf_switch_id += 1

    leaf_half = int(leaf_switch_num / 2)
    assert(leaf_switch_id == leaf_half)

    # Add links between leaf switches and leaf switches.
    for i in range(leaf_half):
        src = leaf_switch_dict[i]
        for j in range(downlink_num):
            dst = NIC_num + leaf_half + i // downlink_num * downlink_num + j
            connect_info_list.append((src, dst, 1))
            connect_info_list.append((dst, src, 1))

    # Add links between spine switches and leaf switches.
    group_num = int(spine_switch_num / downlink_num)
    multiple_link_num = int(downlink_num / group_num)
    for leaf in range(leaf_half, leaf_switch_num):
        src = leaf_switch_dict[leaf] 
        for j in range(group_num):
            dst = NIC_num + leaf_switch_num + leaf % downlink_num * group_num + j
            connect_info_list.append((src, dst, multiple_link_num))
            connect_info_list.append((dst, src, multiple_link_num))

    print('conenction num:', len(connect_info_list))
    return connect_info_list


def create_hw_eps_connect2_list(NIC_num, leaf_switch_num, spine_switch_num, leaf_switch_port_num, spine_switch_port_num, NIC_num_in_a_server):
    """
    isolated 2 plane

    The ID of NIC starts from 0.
    The ID of switch starts from the maximum NIC ID + 1.

    Return:
    - connect_info_list: [(src_id, dst_id, link_num)]
    """

    NPU_num = int(NIC_num / 2)
    NPU_num_in_a_server = int(NIC_num_in_a_server / 2)
    leaf_switch_num_in_a_plane = int(leaf_switch_num / 2)
    spine_switch_num_in_a_plane = int(spine_switch_num / 2)

    connect_info_list = []

    # {key(switch id): value(global device id)} 
    plane1_leaf_switch_dict = {}
    plane2_leaf_switch_dict = {}
    plane1_spine_switch_dict = {}
    plane2_spine_switch_dict = {}
    for i in range(leaf_switch_num_in_a_plane):
        plane1_leaf_switch_dict[i] = i + NIC_num
        plane2_leaf_switch_dict[i] = i + NIC_num + leaf_switch_num_in_a_plane + spine_switch_num_in_a_plane

    for i in range(spine_switch_num_in_a_plane):
        plane1_spine_switch_dict[i] = i + NIC_num + leaf_switch_num_in_a_plane
        plane2_spine_switch_dict[i] = i + NIC_num + leaf_switch_num_in_a_plane + spine_switch_num_in_a_plane + leaf_switch_num_in_a_plane

    plane1_NIC_id_dict = {}
    plane2_NIC_id_dict = {}
    plane1_i = 0
    plane2_i = 0
    for i in range(NIC_num):
        if i // NPU_num_in_a_server % 2 == 0:
            plane1_NIC_id_dict[plane1_i] = i
            plane1_i += 1
        else:
            plane2_NIC_id_dict[plane2_i] = i
            plane2_i += 1

    # Add links between NICs and leaf switches.
    leaf_switch_id = 0
    downlinks = int(leaf_switch_port_num / 2)
    for connected_NIC_id in range(NPU_num):
        # plane1
        connect_info_list.append((plane1_NIC_id_dict[connected_NIC_id], plane1_leaf_switch_dict[leaf_switch_id], 1))
        connect_info_list.append((plane1_leaf_switch_dict[leaf_switch_id], plane1_NIC_id_dict[connected_NIC_id], 1))
        # plane 2
        connect_info_list.append((plane2_NIC_id_dict[connected_NIC_id], plane2_leaf_switch_dict[leaf_switch_id], 1))
        connect_info_list.append((plane2_leaf_switch_dict[leaf_switch_id], plane2_NIC_id_dict[connected_NIC_id], 1))
        if connected_NIC_id > 0 and (connected_NIC_id + 1) % downlinks == 0:
            leaf_switch_id += 1
    
    # Add links between leaf switches and leaf switches.
    pod_num = int(NPU_num / (spine_switch_num_in_a_plane * downlinks))
    for i in range(pod_num):
        for j in range(downlinks):
            plane1_src = plane1_leaf_switch_dict[i * downlinks + j]
            plane2_src = plane2_leaf_switch_dict[i * downlinks + j]
            for k in range(downlinks):
                plane1_dst = plane1_leaf_switch_dict[i * downlinks + j + leaf_switch_num_in_a_plane // 2]
                plane2_dst = plane2_leaf_switch_dict[i * downlinks + j + leaf_switch_num_in_a_plane // 2]
                connect_info_list.append((plane1_src, plane1_dst, 1))
                connect_info_list.append((plane1_dst, plane1_src, 1))
                connect_info_list.append((plane2_src, plane2_dst, 1))
                connect_info_list.append((plane2_dst, plane2_src, 1))

    # Add links between spine switches and leaf switches.
    spine_num_in_a_section = int(spine_switch_num_in_a_plane / downlinks)
    for i in range(pod_num):
        for j in range(downlinks):
            plane1_src = plane1_leaf_switch_dict[i * downlinks + j + leaf_switch_num_in_a_plane // 2]
            plane2_src = plane2_leaf_switch_dict[i * downlinks + j + leaf_switch_num_in_a_plane // 2]
            for k in range(spine_num_in_a_section):
                plane1_dst = plane1_spine_switch_dict[i * spine_num_in_a_section + k]
                plane2_dst = plane2_spine_switch_dict[i * spine_num_in_a_section + k]
                connect_info_list.append((plane1_src, plane1_dst, 8))
                connect_info_list.append((plane1_dst, plane1_src, 8))
                connect_info_list.append((plane2_src, plane2_dst, 8))
                connect_info_list.append((plane2_dst, plane2_src, 8))
    print('conenction num:', len(connect_info_list))
    return connect_info_list


def create_old_clos_connect_list(NIC_num, leaf_switch_num, spine_switch_num, leaf_switch_port_num, spine_switch_port_num, NIC_num_in_a_server):
    """
    The ID of NIC starts from 0.
    The ID of switch starts from the maximum NIC ID + 1.

    Return:
    - connect_info_list: [(src_id, dst_id, link_num)]
    """

    assert(NIC_num == int(leaf_switch_port_num / 2) * leaf_switch_num)
    assert(spine_switch_port_num >= leaf_switch_num)

    connect_info_list = []

    # {key(switch id): value(global device id)} 
    leaf_switch_dict = {}
    spine_switch_dict = {}
    for i in range(leaf_switch_num):
        leaf_switch_dict[i] = i + NIC_num

    for i in range(spine_switch_num):
        spine_switch_dict[i] = i + NIC_num + leaf_switch_num
    
    # Add links between NICs and leaf switches.
    # ---old---
    # connected_NIC_id = 0
    # for leaf_switch_id in range(leaf_switch_num):
    #     for _ in range(downlink_num):
    #         connect_info_list.append((connected_NIC_id, leaf_switch_dict[leaf_switch_id], 1))
    #         connect_info_list.append((leaf_switch_dict[leaf_switch_id], connected_NIC_id, 1))
    #         connected_NIC_id += 1
    # ------
    # server_num = int(NIC_num / NIC_num_in_a_server)
    connected_NIC_id = 0
    for leaf_switch_id in range(0, leaf_switch_num, NIC_num_in_a_server):
        for _ in range(int(leaf_switch_port_num / 2)):
            for i in range(NIC_num_in_a_server):
                connect_info_list.append((connected_NIC_id, leaf_switch_dict[leaf_switch_id + i], 1))
                connect_info_list.append((leaf_switch_dict[leaf_switch_id + i], connected_NIC_id, 1))
                connected_NIC_id += 1

    # Add links between spine switches and leaf switches.
    for leaf_switch_id in range(leaf_switch_num):
        for spine_switch_id in range(spine_switch_num):
            # int(downlink_num / spine_switch_num) can guarantee 1:1 oversubscription.
            connect_info_list.append((spine_switch_dict[spine_switch_id], leaf_switch_dict[leaf_switch_id], int(leaf_switch_port_num / 2 / spine_switch_num)))
            connect_info_list.append((leaf_switch_dict[leaf_switch_id], spine_switch_dict[spine_switch_id], int(leaf_switch_port_num / 2 / spine_switch_num)))

    print('conenction num:', len(connect_info_list))
    return connect_info_list
