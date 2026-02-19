import numpy as np


class CijtSolver:
    """
    该类用cpp实现，用于求解C_ij拆分为C_ijt的问题。
    """
    def __init__(self,
                 c_ij: np.ndarray[int],
                 spine_num_per_pod: int,
                 leaf_num: int):
        """
        :param c_ij: 二维矩阵，表示pod之间的流量需求。
        :param spine_num_per_pod: 每个pod的spine交换机数量。
        :param leaf_num: spine交换机上行端口数量。（实际上leaf交换机的数量就是spine下行
        端口的数量，同时也就等于spine交换机上行端口的数量）
        """
    def solve(self) -> np.ndarray[int]: ...