import numpy as np
from typing import Dict, List, Tuple


class RoutingSolver:
    """
    This class is used to generate the routing table for the OCS expander network.
    """

    def __init__(
            self,
            pod_num: int,
            spine_num: int,
            nic_num: int,
            server_num_per_pod: int,
            spine_up_port_num: int,
            leaf_spine_link_num: int,
            init_alpha: float,
            is_rail_optimized: bool,
    ) -> None:
        """
        Parameters
        ----------
        pod_num : int
            Number of pods in the network.
        spine_num : int
            Number of spine switches in the network.
        nic_num : int
            Number of NICs in the network, also the number of gpus.
        server_num_per_pod : int
            Number of servers in each pod.
        spine_up_port_num : int
            Number of up ports in each spine switch. Also the number of down
            ports in each leaf switch, and the number of leaf switches in each pod.
        init_alpha : float
            The extra weight of initial direct routing path in the weighted routing table.
            When calculating the routing table, the initial weight of direct path will be
            multiplied by init_alpha.
        is_rail_optimized : bool
            Whether the network is rail-optimized.
        """

    def generate_routing_table(self, x_ijkt: np.ndarray) -> None:
        """
        Generate the routing table for the OCS expander network, according to the
        physical topology `x_ijkt`, including intra-pod routing and inter-pod routing.
        """

    def generate_intra_pod_table(self) -> None:
        """
        Generate the intra-pod routing table. Including intra-pod up and down routing.
        """

    def get_intra_pod_up_table(self) -> Dict[int, List[int]]:
        """
        Get the intra-pod up routing table. The key is the device id, and the value is
        the list of next hop leaf/spine switches' id.
        Intra-pod up routing does not need to specify the destination server id.
        """

    def get_intra_pod_down_table(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Get the intra-pod down routing table. The first key is the device id, the second
        key is the destination nic id, and the value is the list of next hop device id.
        Typically, inner dict is the routing table of the device with first key, the routing
        table's key is the destination nic id, value is the list of all available next hop
        devices' id.
        """

    def get_inter_pod_routing_table(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Get the inter-pod routing table. The first key is the device id, the second key is
        the destination pod id, and the value is the list of next hop spine switches' id.
        """

    def get_inter_pod_weighted_direct_routing_table(self) -> Dict[int, Dict[int, List[Tuple[int, float]]]]:
        """
        Get the inter-pod weighted direct routing table. Weighted_routing_table is used for weighted
        ECMP routing or other routing algorithms that need to consider the weight of each path.
        This routing table only contains paths with one hop between different pods.
        """
    
    def get_inter_pod_weighted_twohop_routing_table(self) -> Dict[int, Dict[int, List[Tuple[int, float]]]]:
        """
        Get the inter-pod weighted 2-hop routing table. Weighted_routing_table is used for weighted
        ECMP routing or other routing algorithms that need to consider the weight of each path.
        This routing table only contains paths with 2 hops between different pods.
        """

    def get_connection_info_list(self) -> List[List[int]]:
        """
        Get the connection information list (allocated link mapping).
        """
