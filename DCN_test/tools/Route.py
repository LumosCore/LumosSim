import numpy as np
from collections import Counter
from .utils import dict_to_numpy, create_submatrices, split_until_limit
import time

class RouteTool:
    @classmethod
    def get_traffic(cls, N, f, traffic_matrix):
        return np.array([[sum(f[i, j_prime, j] * traffic_matrix[i, j_prime] for j_prime in range(N)) + \
                          sum(f[i_prime, j, i] * traffic_matrix[i_prime, j] for i_prime in range(N)) if i != j else 0
                          for j in range(N)] for i in range(N)])

    @classmethod
    def calculate_bandwidth_utilization(cls, N, f, traffic_matrix, n_matrix, bandwidth_matrix):
        """
        计算带宽利用率。

        Args:
            N (int): 节点数量。
            f (np.ndarray): 一个形状为(N, N, N)的三维数组，表示节点之间的路径关系。
            traffic_matrix (dict or np.ndarray): 一个形状为(N, N)的二维数组或字典，表示节点之间的流量矩阵。
            n_matrix (np.ndarray): 一个形状为(N, N)的二维数组，表示节点之间的连接关系。
            bandwidth_matrix (dict or np.ndarray): 一个形状为(N, N)的二维数组或字典，表示节点之间的带宽矩阵。

        Returns:
            np.ndarray: 一个形状为(N, N)的二维数组，表示带宽利用率。

        """
        # Convert n_matrix and bandwidth_matrix to NumPy arrays if they aren't already
        n_matrix = np.asarray(n_matrix)
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)
        numerator = np.zeros((N, N))

        for k in range(N):
            middd = f[:, :, k] * traffic_matrix
            numerator[:, k] += np.sum(middd, axis=1)
            numerator[k, :] += np.sum(middd, axis=0)
        # Compute the denominator
        # import pdb;pdb.set_trace()
        denominator = n_matrix * bandwidth_matrix
        # Avoid division by zero by using np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            utilization = np.where(denominator != 0, numerator / denominator, 0)
        # Handle cases where both numerator and denominator are zero
        utilization = np.where((denominator == 0) & (numerator == 0), 0, utilization)
        return utilization

    @classmethod
    def update_n_matrix_based_on_utilization(cls, N, R, R_c, n_matrix, utilization,T_tmp,d_wave):
        """
        根据带宽利用率更新 n_matrix。
        Args:
            N (int): 节点数量。
            R (list or np.ndarray): 一个长度为 N 的列表或数组，表示每个节点的剩余容量。
            R_c (list or np.ndarray): 一个长度为 N 的列表或数组，表示每个节点的容量。
            n_matrix (np.ndarray): 一个形状为(N, N)的二维数组，表示节点之间的连接关系。
            utilization (np.ndarray): 一个形状为(N, N)的二维数组，表示带宽利用率。
        Returns:
            无返回值。
        """
        flattened_matrix = utilization.flatten()
        sorted_indices = np.argsort(-flattened_matrix)
        sorted_indices_2d = np.unravel_index(sorted_indices, utilization.shape)
        sorted_index_tuples = list(zip(*sorted_indices_2d))
        
        # 创建R_remain的副本，以便在循环中更新
        R_remain = [i - j for i, j in zip(R, R_c)]
        R_remain_array = np.array(R_remain)
        
        # 记录处理的链路数量
        processed_links = 0
        max_links_to_process = 1000000  # 最多处理前20个链路
        
        for index in sorted_index_tuples:
            # 检查是否是自环
            if index[0] == index[1]:
                continue
                
            # 检查是否还有足够的节点有剩余端口
            if np.sum(R_remain_array > 0) <= 1:
                break
                
            # 检查两端节点是否都有剩余端口
            if R_remain_array[index[0]] > 0 and R_remain_array[index[1]] > 0:
                # 计算可以增加的链路数量
                line_plus = min(R_remain_array[index[0]], R_remain_array[index[1]])
                
                # 更新n_matrix和R_c
                n_matrix[index] += line_plus
                n_matrix[index[1], index[0]] += line_plus
                R_c[index[0]] += line_plus
                R_c[index[1]] += line_plus
                
                # 更新R_remain_array
                R_remain_array[index[0]] -= line_plus
                R_remain_array[index[1]] -= line_plus
                
                # 增加处理的链路计数
                processed_links += 1
                
                # 如果已经处理了足够多的链路，就退出
                if processed_links >= max_links_to_process:
                    break

    @classmethod
    def initialize_f(cls, N, n_matrix):
        """
        初始化函数，用于生成一个三维数组f，表示节点之间的路径关系。

        Args:
            无参数。

        Returns:
            f (np.ndarray): 一个形状为(N, N, N)的三维数组，表示节点之间的路径关系。

        Raises:
            无异常抛出。

        """
        f = np.zeros((N, N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i == j:
                    pass
                elif n_matrix[i, j] != 0:
                    for k in range(N):
                        if k == j:
                            f[i, j, k] = 1  # Direct path
                else:
                    found = False
                    for k in range(N):
                        if n_matrix[i, k] != 0 and n_matrix[k, j] != 0 and found == False:
                            f[i, j, k] = 1  # Found a valid intermediate node
                            found = True
        return f

    @classmethod
    def binary_search_optimal_u(cls, utilization, update_f, i, j, traffic, mask_matrix, upper_bound=None,
                                lower_bound=None, target_sum=1,
                                tol=1e-10):
        """
        二分搜索最优的 u，使得 np.sum(update_f(utilization, u)) 等于 target_sum。

        参数:
        utilization (numpy.ndarray): 利用率数组。
        update_f (function): 一个函数，接受 (utilization, u) 作为输入，返回一个与 utilization 相同形状的数组。
        target_sum (float): 目标和，默认为 1。
        tol (float): 容忍度，用于停止搜索的条件，默认为 1e-6。

        返回:
        float: 最优的 u 值。
        """
        if upper_bound is None:
            upper_bound = np.max(utilization)
            lower_bound = np.min(utilization)

        while upper_bound - lower_bound > tol:
            mid_u = (lower_bound + upper_bound) / 2
            if np.min(update_f(utilization, mid_u, i, j, traffic, mask_matrix))<0:
                import pdb;pdb.set_trace()
                
            result = np.sum(update_f(utilization, mid_u, i, j, traffic, mask_matrix))

            if result > target_sum:
                upper_bound = mid_u
            else:
                lower_bound = mid_u

        return (lower_bound + upper_bound) / 2

    @classmethod
    def lp_rapid(cls, n_matrix, bandwidth_matrix, traffic_matrix, N, epoch, f, mask_matrix,time_limit, tol=1e-5):
        """
        通过线性迭代方式快速求解路由矩阵f，并返回f和最大带宽利用率。

        Args:
            n_matrix (np.ndarray): 形状为 (N, N) 的二维数组，表示节点之间的连接关系。
            bandwidth_matrix (dict or np.ndarray): 形状为 (N, N) 的二维数组或字典，表示节点之间的带宽矩阵。
            traffic_matrix (dict or np.ndarray): 形状为 (N, N) 的二维数组或字典，表示节点之间的流量矩阵。
            N (int): 节点总数。
            epoch (int): 迭代次数。
            tol (float, optional): 收敛阈值。默认为1e-5。

        Returns:
            Tuple[np.ndarray, float]: 返回求解得到的路由矩阵f和最大带宽利用率。

        Raises:
            ValueError: 当 traffic_matrix[i, j] 为 0 时，在 update_f 函数内部抛出异常以避免除以零。

        """

        def update_f(utilization, u_max_prime, i, j, demand, mask_matrix):
            if demand[i, j] == 0:
                f_ij = np.zeros(utilization.shape[0])
                return f_ij
            # 计算 f_ij_f 和 f_ij_b
            f_ij_f = (u_max_prime - utilization[i, :]) * bandwidth_matrix[i, :] / \
                     demand[i, j]
            f_ij_b = (u_max_prime - utilization[:, j]) * bandwidth_matrix[:, j] / \
                     demand[i, j]

            # 确保 f_ij_f[i] 为 0，f_ij_f[j] 为正无穷
            f_ij_f[i] = 0
            f_ij_f[j] = float("inf")

            # 计算初始的 f_ij
            f_ij = np.minimum(f_ij_f, f_ij_b)
            f_ij = np.maximum(f_ij, 0)

            # 更新 f_ij[j]
            f_ij[j] = np.maximum((u_max_prime - utilization[i, j]), 0) * bandwidth_matrix[i, j] / \
                      demand[i, j]
            if np.min(f_ij) < 0:
                print(1)
                import pdb;pdb.set_trace()
            # 应用在初始化时计算的掩码矩阵
            f_ij *= mask_matrix[i, j, :]

            return f_ij

        # def update_f(utilization, u_max_prime, i, j):
        #     """
        #     更新 f 矩阵的指定行和列。
        #
        #     Args:
        #         utilization (np.ndarray): 形状为 (N, N) 的二维数组，表示带宽利用率矩阵。
        #         u_max_prime (float): 带宽利用率上限值。
        #         i (int): 需要更新的行索引。
        #         j (int): 需要更新的列索引。
        #
        #     Returns:
        #         np.ndarray: 更新后的 f 矩阵的指定行和列，形状为 (N,) 的一维数组。
        #
        #     Raises:
        #         ValueError: 当 traffic_matrix[i, j] 为 0 时，抛出异常以避免除以零。
        #
        #     """
        #     if traffic_matrix[i, j] == 0:
        #         raise ValueError("traffic_matrix[i, j] cannot be zero to avoid division by zero.")
        #     f_ij_f = (u_max_prime - utilization[i, :]) * n_matrix[i, :] * bandwidth_matrix[i, :] / traffic_matrix[i, j]
        #     f_ij_b = (u_max_prime - utilization[:, j]) * n_matrix[:, j] * bandwidth_matrix[:, j] / traffic_matrix[i, j]
        #     f_ij_f[i] = 0
        #     f_ij_f[j] = float("inf")
        #     f_ij = np.minimum(f_ij_f, f_ij_b)
        #     f_ij = np.maximum(f_ij, 0)
        #     f_ij[j] = (u_max_prime - utilization[i, j]) * n_matrix[i, j] * bandwidth_matrix[i, j] / traffic_matrix[i, j]
        #     return f_ij

        def get_start_end_array(max_indices, N):
            """
            根据最大索引数组构建所有需要优化的通信对，并返回通信对列表。

            Args:
                max_indices (np.ndarray): 一个二维数组，每行表示一个最大索引，包含两个元素：起点和终点。
                N (int): 节点总数。

            Returns:
                list: 所有需要优化的通信对列表，每个通信对是一个包含两个整数的元组，表示起点和终点。

            """
            # 构建所有需要优化的通信对，存在重复通信对
            all_points = set(range(N))
            starts = max_indices[:, 0]
            ends = max_indices[:, 1]
            start_counts = Counter(starts)
            end_counts = Counter(ends)
            total_counts = start_counts + end_counts
            # 合并起点和终点信息，并添加是否是起点的标记
            points_info = []
            for point, count in start_counts.items():
                points_info.append((count, point, True))  # True 表示是起点
            for point, count in end_counts.items():
                if point not in [p for _, p, _ in points_info]:  # 避免重复添加同一点
                    points_info.append((count, point, False))  # False 表示是终点
            # 按频率排序（降序），如果频率相同，则按是否是起点排序（可以调整以符合你的具体需求）
            points_info.sort(key=lambda x: (-x[0], x[2]), reverse=True)
            # 创建一个新的字典，包含所有点及其总次数（未出现的点计为0）
            sorted_points_dict = {point: total_counts.get(point, 0) for point in all_points}
            # 将字典转换为列表，并按总次数降序排序
            sorted_points = sorted(sorted_points_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_points = sorted_points_dict.items()
            sorted_points = [i[0] for i in sorted_points]
            pairs = []
            # 遍历 points_info 中的每个点信息
            for info in points_info:
                _, point, is_start = info  # 解包点、频率（这里未使用）和是否是起点
                # 生成与其他所有点的配对（不包括自身）
                for other_point in sorted_points:
                    if other_point != point:  # 确保不与自己配对
                        if is_start:
                            
                            pairs.append((int(point), int(other_point)))  # 起点在前
                        else:
                            pairs.append((int(other_point), int(point)))  # 终点在后（如果需要的话）
            return pairs
        start_time=time.time()
        # f = cls.initialize_f(N, n_matrix)
        mask = (bandwidth_matrix == 0)
        # import pdb;pdb.set_trace()
        # # 获取 bandwidth_matrix 中值为 0 的位置
        # mask = (bandwidth_matrix == 0)
        mask_matrix=np.copy(mask_matrix)
        # 对 mask_matrix 进行操作
        for i in range(mask_matrix.shape[0]):
            # 对于每个 i，找出 bandwidth_matrix[i,j] == 0 的所有 j
            j_zeros = np.where(mask[i])[0]
            
            # 将 mask_matrix[i,:,j] 设置为 0
            mask_matrix[i, :, j_zeros] = 0
        for j in range(mask_matrix.shape[0]):
            # 对于每个 i，找出 bandwidth_matrix[i,j] == 0 的所有 j
            i_zeros = np.where(mask[j])[0]
            # import pdb;pdb.set_trace()
            # 将 mask_matrix[i,:,j] 设置为 0
            mask_matrix[:, i_zeros, j] = 0    
        for i in range(mask_matrix.shape[0]):
            for j in range(mask_matrix.shape[0]):
                if bandwidth_matrix[i,j]>0:
                    mask_matrix[i, j, j] = 1 
        # import pdb;pdb.set_trace()
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)

        utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        utilization[mask] = 0
        opt = np.max(utilization)

        for count in range(epoch):
            # print("搜索一轮")
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            # print(f"搜索一轮{max_value}")
           
            pairs = get_start_end_array(np.argwhere(np.round(utilization, 6) == max_value),utilization.shape[0])
            # pairs = get_start_end_array(max_indices, N)
            # import pdb;pdb.set_trace()
            for i, j in pairs:
                
                # print("搜索一对")
                if traffic_matrix[i, j] <= 1e-5:
                    continue
                no_i = (np.arange(N) != i) & ~mask[:,i ]
                no_j = (np.arange(N) != j) & ~mask[:, j]

                if np.min(bandwidth_matrix[no_j, j]) == 0 or np.min(bandwidth_matrix[no_i, i]) == 0:
                    print(1)
                    import pdb;pdb.set_trace()

                utilization[i, no_i] = utilization[i, no_i] - traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] - traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])
                # utilization[mask] = 0
                u_max_prime = max(np.max(utilization[i, no_i]), np.max(utilization[no_j, j]))
                f_ij = update_f(utilization, u_max_prime, i, j, traffic_matrix, mask_matrix)
                if f_ij.sum() >= 1:
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            tol=tol)
                else:
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            upper_bound=opt,
                                                            lower_bound=u_max_prime, target_sum=1, tol=tol)
                f_ij = update_f(utilization, optimal_u, i, j, traffic_matrix, mask_matrix)
                if f_ij.sum() == 0:
                    import pdb;pdb.set_trace()
                f_ij = f_ij / f_ij.sum()
                f[i, j, :] = f_ij
                utilization[i, no_i] = utilization[i, no_i] + traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] + traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])

                # utilization[mask] = 0

            if opt - np.max(utilization) <= tol * 1e-1:
                break
            if time.time()-start_time >= time_limit:
                break
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)
        # import pdb;pdb.set_trace()
        # utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        # utilization[mask] = 0
        # max_value = np.max(utilization)
        # max_utilization_history.append(np.max(utilization))
        # time_history.append(current_time)
        return max_value, f


    @classmethod
    def static_lp_rapid(cls, n_matrix, bandwidth_matrix, traffic_matrix, N, epoch, f, mask_matrix,time_limit, tol=1e-5):
        """
        通过线性迭代方式快速求解路由矩阵f，并返回f和最大带宽利用率。

        Args:
            n_matrix (np.ndarray): 形状为 (N, N) 的二维数组，表示节点之间的连接关系。
            bandwidth_matrix (dict or np.ndarray): 形状为 (N, N) 的二维数组或字典，表示节点之间的带宽矩阵。
            traffic_matrix (dict or np.ndarray): 形状为 (N, N) 的二维数组或字典，表示节点之间的流量矩阵。
            N (int): 节点总数。
            epoch (int): 迭代次数。
            tol (float, optional): 收敛阈值。默认为1e-5。

        Returns:
            Tuple[np.ndarray, float]: 返回求解得到的路由矩阵f和最大带宽利用率。

        Raises:
            ValueError: 当 traffic_matrix[i, j] 为 0 时，在 update_f 函数内部抛出异常以避免除以零。

        """

        def update_f(utilization, u_max_prime, i, j, demand, mask_matrix):
            if demand[i, j] == 0:
                f_ij = np.zeros(utilization.shape[0])
                return f_ij
            # 计算 f_ij_f 和 f_ij_b
            f_ij_f = (u_max_prime - utilization[i, :]) * bandwidth_matrix[i, :] / \
                     demand[i, j]
            f_ij_b = (u_max_prime - utilization[:, j]) * bandwidth_matrix[:, j] / demand[i, j]

            # 确保 f_ij_f[i] 为 0，f_ij_f[j] 为正无穷
            f_ij_f[i] = 0
            f_ij_f[j] = float("inf")

            # 计算初始的 f_ij
            f_ij = np.minimum(f_ij_f, f_ij_b)
            f_ij = np.maximum(f_ij, 0)

            # 更新 f_ij[j]
            f_ij[j] = np.maximum((u_max_prime - utilization[i, j]), 0) * bandwidth_matrix[i, j] / \
                      demand[i, j]
            if np.min(f_ij) < 0:
                print(1)
                import pdb;pdb.set_trace()
            # 应用在初始化时计算的掩码矩阵
            f_ij *= mask_matrix[i, j, :]

            return f_ij

        # def update_f(utilization, u_max_prime, i, j):
        #     """
        #     更新 f 矩阵的指定行和列。
        #
        #     Args:
        #         utilization (np.ndarray): 形状为 (N, N) 的二维数组，表示带宽利用率矩阵。
        #         u_max_prime (float): 带宽利用率上限值。
        #         i (int): 需要更新的行索引。
        #         j (int): 需要更新的列索引。
        #
        #     Returns:
        #         np.ndarray: 更新后的 f 矩阵的指定行和列，形状为 (N,) 的一维数组。
        #
        #     Raises:
        #         ValueError: 当 traffic_matrix[i, j] 为 0 时，抛出异常以避免除以零。
        #
        #     """
        #     if traffic_matrix[i, j] == 0:
        #         raise ValueError("traffic_matrix[i, j] cannot be zero to avoid division by zero.")
        #     f_ij_f = (u_max_prime - utilization[i, :]) * n_matrix[i, :] * bandwidth_matrix[i, :] / traffic_matrix[i, j]
        #     f_ij_b = (u_max_prime - utilization[:, j]) * n_matrix[:, j] * bandwidth_matrix[:, j] / traffic_matrix[i, j]
        #     f_ij_f[i] = 0
        #     f_ij_f[j] = float("inf")
        #     f_ij = np.minimum(f_ij_f, f_ij_b)
        #     f_ij = np.maximum(f_ij, 0)
        #     f_ij[j] = (u_max_prime - utilization[i, j]) * n_matrix[i, j] * bandwidth_matrix[i, j] / traffic_matrix[i, j]
        #     return f_ij

        def get_start_end_array(max_indices, N):
            """
            根据最大索引数组构建所有需要优化的通信对，并返回通信对列表。

            Args:
                max_indices (np.ndarray): 一个二维数组，每行表示一个最大索引，包含两个元素：起点和终点。
                N (int): 节点总数。

            Returns:
                list: 所有需要优化的通信对列表，每个通信对是一个包含两个整数的元组，表示起点和终点。

            """
            # 构建所有需要优化的通信对，存在重复通信对
            all_points = set(range(N))
            starts = max_indices[:, 0]
            ends = max_indices[:, 1]
            start_counts = Counter(starts)
            end_counts = Counter(ends)
            total_counts = start_counts + end_counts
            # 合并起点和终点信息，并添加是否是起点的标记
            points_info = []
            for point, count in start_counts.items():
                points_info.append((count, point, True))  # True 表示是起点
            for point, count in end_counts.items():
                if point not in [p for _, p, _ in points_info]:  # 避免重复添加同一点
                    points_info.append((count, point, False))  # False 表示是终点
            # 按频率排序（降序），如果频率相同，则按是否是起点排序（可以调整以符合你的具体需求）
            points_info.sort(key=lambda x: (-x[0], x[2]), reverse=True)
            # 创建一个新的字典，包含所有点及其总次数（未出现的点计为0）
            sorted_points_dict = {point: total_counts.get(point, 0) for point in all_points}
            # 将字典转换为列表，并按总次数降序排序
            sorted_points = sorted(sorted_points_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_points = sorted_points_dict.items()
            sorted_points = [i[0] for i in sorted_points]
            pairs = []
            # 遍历 points_info 中的每个点信息
            for info in points_info:
                _, point, is_start = info  # 解包点、频率（这里未使用）和是否是起点
                # 生成与其他所有点的配对（不包括自身）
                for other_point in sorted_points:
                    if other_point != point:  # 确保不与自己配对
                        if is_start:
                            
                            pairs.append((int(point), int(other_point)))  # 起点在前
                        else:
                            pairs.append((int(other_point), int(point)))  # 终点在后（如果需要的话）
            return pairs
        start_time=time.time()
        # f = cls.initialize_f(N, n_matrix)
        mask = (bandwidth_matrix == 0)
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)
        utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        utilization[mask] = 0
        opt = np.max(utilization)
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            # nonzero_coords = np.nonzero(traffic_matrix)
            # pairs = list(zip(nonzero_coords[0], nonzero_coords[1]))
            # pairs = get_start_end_array(np.argwhere(np.round(utilization, 6) == max_value),utilization.shape[0])
            pairs = get_start_end_array(max_indices, N)
            for i, j in pairs:
                if traffic_matrix[i, j] == 0:
                    continue
                no_i = (np.arange(N) != i) & ~mask[i, :]
                no_j = (np.arange(N) != j) & ~mask[:, j]

                if np.min(bandwidth_matrix[no_j, j]) == 0 or np.min(bandwidth_matrix[no_i, i]) == 0:
                    print(1)
                    import pdb;pdb.set_trace()

                utilization[i, no_i] = utilization[i, no_i] - traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] - traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])
                # utilization[mask] = 0
                u_max_prime = np.max(utilization)
                f_ij = update_f(utilization, u_max_prime, i, j, traffic_matrix, mask_matrix)
                if sum(f_ij)<=1:
                    optimal_u =u_max_prime
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            upper_bound=opt,
                                                            lower_bound=np.max(utilization), target_sum=1, tol=tol)
                else:
                    optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                                                            upper_bound=np.max(utilization),
                                                            lower_bound=0, target_sum=1, tol=tol)
                    
                
                f_ij = update_f(utilization, optimal_u, i, j, traffic_matrix, mask_matrix)
                if f_ij.sum()==0:
                    import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
                f_ij = f_ij / f_ij.sum()
                f[i, j, :] = f_ij
                utilization[i, no_i] = utilization[i, no_i] + traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] + traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])

                # utilization[mask] = 0

            if opt - np.max(utilization) <= tol * 1e-1:
                break
            if time.time()-start_time >= time_limit:
                break
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)
        # utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        # utilization[mask] = 0
        # max_value = np.max(utilization)
        # max_utilization_history.append(np.max(utilization))
        # time_history.append(current_time)
        return max_value, f,max_utilization_history,time_history
    
    @classmethod
    def lp_lp_rapid(cls, n_matrix, bandwidth_matrix, traffic_matrix, N, epoch, f, mask_matrix,time_limit, tol=1e-5):
        """
        通过线性迭代方式快速求解路由矩阵f，并返回f和最大带宽利用率。

        Args:
            n_matrix (np.ndarray): 形状为 (N, N) 的二维数组，表示节点之间的连接关系。
            bandwidth_matrix (dict or np.ndarray): 形状为 (N, N) 的二维数组或字典，表示节点之间的带宽矩阵。
            traffic_matrix (dict or np.ndarray): 形状为 (N, N) 的二维数组或字典，表示节点之间的流量矩阵。
            N (int): 节点总数。
            epoch (int): 迭代次数。
            tol (float, optional): 收敛阈值。默认为1e-5。

        Returns:
            Tuple[np.ndarray, float]: 返回求解得到的路由矩阵f和最大带宽利用率。

        Raises:
            ValueError: 当 traffic_matrix[i, j] 为 0 时，在 update_f 函数内部抛出异常以避免除以零。

        """

        def update_f(utilization, u_max_prime, i, j, demand, mask_matrix):
            if demand[i, j] == 0:
                f_ij = np.zeros(utilization.shape[0])
                return f_ij
            # 计算 f_ij_f 和 f_ij_b
            f_ij_f = (u_max_prime - utilization[i, :]) * bandwidth_matrix[i, :] / demand[i, j]
            f_ij_b = (u_max_prime - utilization[:, j]) * bandwidth_matrix[:, j] / demand[i, j]

            # 确保 f_ij_f[i] 为 0，f_ij_f[j] 为正无穷
            f_ij_f[i] = 0
            f_ij_f[j] = float("inf")

            # 计算初始的 f_ij
            f_ij = np.minimum(f_ij_f, f_ij_b)
            # f_ij = np.maximum(f_ij, 0)

            # 更新 f_ij[j]
            f_ij[j] = (u_max_prime - utilization[i, j]) * bandwidth_matrix[i, j] / \
                      demand[i, j]
            if np.min(f_ij) < -1e-10:
                print(1)
                import pdb;pdb.set_trace()
            # 应用在初始化时计算的掩码矩阵
            f_ij *= mask_matrix[i, j, :]

            return f_ij

 
        def get_start_end_array(max_indices, N):
            """
            根据最大索引数组构建所有需要优化的通信对，并返回通信对列表。

            Args:
                max_indices (np.ndarray): 一个二维数组，每行表示一个最大索引，包含两个元素：起点和终点。
                N (int): 节点总数。

            Returns:
                list: 所有需要优化的通信对列表，每个通信对是一个包含两个整数的元组，表示起点和终点。

            """
            # 构建所有需要优化的通信对，存在重复通信对
            all_points = set(range(N))
            starts = max_indices[:, 0]
            ends = max_indices[:, 1]
            start_counts = Counter(starts)
            end_counts = Counter(ends)
            total_counts = start_counts + end_counts
            # 合并起点和终点信息，并添加是否是起点的标记
            points_info = []
            for point, count in start_counts.items():
                points_info.append((count, point, True))  # True 表示是起点
            for point, count in end_counts.items():
                if point not in [p for _, p, _ in points_info]:  # 避免重复添加同一点
                    points_info.append((count, point, False))  # False 表示是终点
            # 按频率排序（降序），如果频率相同，则按是否是起点排序（可以调整以符合你的具体需求）
            points_info.sort(key=lambda x: (-x[0], x[2]), reverse=True)
            # 创建一个新的字典，包含所有点及其总次数（未出现的点计为0）
            sorted_points_dict = {point: total_counts.get(point, 0) for point in all_points}
            # 将字典转换为列表，并按总次数降序排序
            sorted_points = sorted(sorted_points_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_points = sorted_points_dict.items()
            sorted_points = [i[0] for i in sorted_points]
            pairs = []
            # 遍历 points_info 中的每个点信息
            for info in points_info:
                _, point, is_start = info  # 解包点、频率（这里未使用）和是否是起点
                # 生成与其他所有点的配对（不包括自身）
                for other_point in sorted_points:
                    if other_point != point:  # 确保不与自己配对
                        if is_start:
                            
                            pairs.append((int(point), int(other_point)))  # 起点在前
                        else:
                            pairs.append((int(other_point), int(point)))  # 终点在后（如果需要的话）
            return pairs
        start_time=time.time()
        # f = cls.initialize_f(N, n_matrix)
        mask = (bandwidth_matrix == 0)
        bandwidth_matrix = dict_to_numpy(bandwidth_matrix, N)
        traffic_matrix = dict_to_numpy(traffic_matrix, N)
        utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        utilization[mask] = 0
        opt = np.max(utilization)
        max_utilization_history = []
        time_history = []
        for count in range(epoch):
            rounded_utilization = np.round(utilization, 6)
            max_value = np.max(rounded_utilization)
            max_indices = np.argwhere(rounded_utilization == max_value)
            pairs = get_start_end_array(np.argwhere(np.round(utilization, 6) == max_value),utilization.shape[0])
            # pairs = get_start_end_array(max_indices, N)
            for i, j in pairs:
                if traffic_matrix[i, j] == 0:
                    continue
                no_i = (np.arange(N) != i) & ~mask[i, :]
                no_j = (np.arange(N) != j) & ~mask[:, j]



                utilization[i, no_i] = utilization[i, no_i] - traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] - traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])
                # utilization[mask] = 0

                # u_max_prime0 = np.max(utilization)
                # f_ij2 = update_f(utilization, u_max_prime0, i, j, traffic_matrix, mask_matrix)
                # optimal_u = cls.binary_search_optimal_u(utilization, update_f, i, j, traffic_matrix, mask_matrix,
                #                                             upper_bound=opt,
                #                                             lower_bound=u_max_prime0, target_sum=1, tol=tol)
                # 从 Gurobi 求解路径分流比
                f_ij0, u_max_prime = cls.update_f_with_lp(i, j, traffic_matrix, bandwidth_matrix, utilization, mask_matrix, N)
                # if  abs(optimal_u-u_max_prime)>=1e-5:
                #     import pdb;pdb.set_trace()
                # f_ij = update_f(utilization, u_max_prime, i, j, traffic_matrix, mask_matrix)
                # print(u_max_prime,optimal_u)
                # import pdb;pdb.set_trace()
                # 更新 f 和链路利用率
                # f_ij = f_ij / f_ij.sum()
                # if np.sum(f_ij2)<=1:
                #     import pdb;pdb.set_trace()
                    
                f[i, j, :] = f_ij0
                
    
                
                utilization[i, no_i] = utilization[i, no_i] + traffic_matrix[i, j] * f[i, j, no_i] / (
                        n_matrix[i, no_i] * bandwidth_matrix[i, no_i])
                utilization[no_j, j] = utilization[no_j, j] + traffic_matrix[i, j] * f[i, j, no_j] / (
                        n_matrix[no_j, j] * bandwidth_matrix[no_j, j])

                if time.time()-start_time >= time_limit:
                    print("time limit1")
                    break 
                    
                


                # utilization[mask] = 0

            if opt - np.max(utilization) <= tol * 1e-1:
                break
            if time.time()-start_time >= time_limit:
                print("time limit2")
                break
            opt = np.max(utilization)
            # current_time = time.time() - start_time
            # max_utilization_history.append(opt)
            # time_history.append(current_time)
        # utilization = cls.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
        # utilization[mask] = 0
        # max_value = np.max(utilization)

        # max_utilization_history.append(np.max(utilization))
        # time_history.append(current_time)
        return max_value, f,max_utilization_history,time_history
    @classmethod
    def update_f_with_lp(cls, i, j, traffic_matrix, bandwidth_matrix, utilization, mask_matrix, N):
        """
        使用线性规划优化路径分流比 f[i, j, k]，同时最小化全局 MLU。

        Args:
            i, j: 当前的 SD 对。
            traffic_matrix: 当前流量需求矩阵。
            bandwidth_matrix: 带宽矩阵。
            utilization: 当前链路利用率矩阵。
            mask_matrix: 掩码矩阵，用于指示哪些路径可用。
            N: 节点数量。

        Returns:
            f_ij: 更新后的分流比矩阵。
            u_max: 优化后的最大链路利用率。
        """
        from gurobipy import Model, GRB, quicksum

        # 初始化 Gurobi 模型
        m = Model("update_f")
        m.Params.OutputFlag = 0  # 禁止日志输出

        # 定义变量
        f = m.addVars(N, lb=0, ub=1, name="f")  # 分流比
        u = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="u")  # 最大链路利用率

        # 目标函数：最小化全局最大链路利用率
        m.setObjective(u, GRB.MINIMIZE)

        # 约束 1：路径分流比总和为 1
        m.addConstr(quicksum(f[k] for k in range(N)) == 1, name="sum_f")

        # 约束 2：链路利用率限制
        for k in range(N):
            if mask_matrix[i, j, k] == 0:
                m.addConstr(f[k] == 0, name=f"mask_{i}_{j}_{k}")
            else:
                # 动态生成路径的边
                if j==k:
                    edges=[(i,j)]
                elif i==k:
                    m.addConstr(
                        f[k] == 0,
                        name=f"util_{i}_{j}_{k}_{edge}"
                    )
                    edges=[]
                
                else:
                    edges = [(i,k),(k,j)]
                    
                
                for edge in edges:
                    edge_utilization = utilization[edge[0], edge[1]]
                    
                    capacity = bandwidth_matrix[edge[0], edge[1]]
                    if capacity==0:
                        import pdb; pdb.set_trace()
                    # 添加约束：当前路径上的链路利用率不得超过 u
                    m.addConstr(
                        f[k] * traffic_matrix[i, j]/capacity + edge_utilization <= u,
                        name=f"util_{i}_{j}_{k}_{edge}"
                    )

        # 约束 3：u >= 当前所有链路的利用率
        for x in range(N):
            for y in range(N):
                if bandwidth_matrix[x, y] > 0:
                    m.addConstr(
                        u >= utilization[x, y],  # u 必须大于或等于当前的链路利用率
                        name=f"u_ge_utilization_{x}_{y}"
                    )

        # 优化
        m.optimize()

        # 提取结果
        if m.Status == GRB.OPTIMAL:
            f_ij = np.array([f[k].X if mask_matrix[i, j, k] == 1 else 0 for k in range(N)])
            u_max = u.X
        else:
            print("No feasible solution found.")
            f_ij = np.zeros(N)
            u_max = np.max(utilization)

        return f_ij, u_max
    @classmethod
    def lp_by_gp(cls, n_matrix, s_matrix, d_wave, N, mask_matrix):
        from gurobipy import Model, GRB, quicksum
        f_d = {}
        # 初始化模型
        m = Model("NetworkOptimization")
        # 变量
        f = m.addVars(N, N, N, name="f", lb=0)  # 流量分配变量
        u = m.addVar(name="u")  # 目标变量

        # 目标函数
        m.setObjective(u, GRB.MINIMIZE)

        # 约束条件（省略了部分与原始代码相同的约束）
        # 约束

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if i == j or i == k:
                        m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
                    # if mask_matrix[i,j,k]==0:
                    #     m.addConstr(f[i, j, k] == 0, name=f"f_zero_{i}_{j}_{k}")
                        

        # 流量分配总和约束
        for i in range(N):
            for j in range(N):
                if i != j:
                    m.addConstr(quicksum(f[i, j, k] for k in range(N)) == 1, name=f"flow_sum_{i}_{j}")

        # 成本约束（优化后的版本，减少了重复计算）
        cost_sums = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    outgoing_cost = quicksum(f[i, jp, j] * d_wave[i, jp] for jp in range(N))
                    incoming_cost = quicksum(f[ip, j, i] * d_wave[ip, j] for ip in range(N))
                    cost_sums[(i, j)] = outgoing_cost + incoming_cost
                    m.addConstr(cost_sums[(i, j)] <= u * n_matrix[i][j] * s_matrix[i, j], name=f"cost_{i}_{j}")

        # 模型求解
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 43200 )
        m.optimize()
        f_d = np.zeros((N, N, N))
        # 输出结果
        if m.SolCount > 0:
            for key, var in f.items():
                f_d[key] = var.x
            
 
            return u.x, f_d
        else:
            print("No feasible solution found within the time limit.")
            return 100, 100

    @classmethod
    def lp_by_gp_sub(cls, n_matrix, s_matrix, d_wave, N, mask_matrix):
        from gurobipy import Model, GRB, quicksum
        f_d = {}
        # 初始化模型
        m = Model("NetworkOptimization")
        non_zero_indices = np.argwhere(d_wave != 0)
        # 变量
        name_path_weight = [f'w_{i}_{j}_{k}'
                            for i, j in non_zero_indices
                            for k in range(N)
                            ]
        f = m.addVars(name_path_weight, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='path_weight')
        # f = m.addVars(N, N, N, name="f", lb=0)  # 流量分配变量
        u = m.addVar(name="u")  # 目标变量

        # 目标函数
        m.setObjective(u, GRB.MINIMIZE)

        # 约束条件（省略了部分与原始代码相同的约束）
        # 约束

        for i, j in non_zero_indices:
            for k in range(N):
                if i == j or i == k:
                    m.addConstr(f[f'w_{i}_{j}_{k}'] == 0, name=f"f_zero_{i}_{j}_{k}")

        # 流量分配总和约束
        for i, j in non_zero_indices:
            if i != j:
                m.addConstr(quicksum(f[f'w_{i}_{j}_{k}'] for k in range(N)) == 1, name=f"flow_sum_{i}_{j}")

        # 成本约束（优化后的版本，减少了重复计算）
        cost_sums = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    outgoing_cost = quicksum(f[f'w_{i}_{jp}_{j}'] * d_wave[i, jp] for jp in range(N) if
                                             np.any(np.all(non_zero_indices == [i, jp], axis=1)))
                    incoming_cost = quicksum(f[f'w_{ip}_{j}_{i}'] * d_wave[ip, j] for ip in range(N) if
                                             np.any(np.all(non_zero_indices == [ip, j], axis=1)))
                    cost_sums[(i, j)] = outgoing_cost + incoming_cost
                    m.addConstr(cost_sums[(i, j)] <= u * n_matrix[i][j] * s_matrix[i, j], name=f"cost_{i}_{j}")

        # 模型求解
        m.setParam('OutputFlag', 0)
        m.optimize()

        import re
        pattern = r'w_(\d+)_(\d+)_(\d+)'
        # 提取 i, j, k
        extracted_values = [re.match(pattern, name).groups() for name in name_path_weight]

        # 将字符串转换为整数
        extracted_values = [(int(i), int(j), int(k)) for i, j, k in extracted_values]
        f_d = np.zeros((N, N, N))
        # 输出结果
        if m.status == GRB.OPTIMAL:
            index = 0
            for key, var in f.items():
                f_d[extracted_values[index]] = var.x
                index += 1
            return f_d, u.x
        else:
            print("error")

    # @classmethod
    # def lp_by_pop(cls, n_matrix, s_matrix, d_wave, N, pop_number):
    #     import time
    #     s_matrix = dict_to_numpy(s_matrix, N)
    #     d_wave = dict_to_numpy(d_wave, N)

    #     d_pop_list = create_submatrices(d_wave, pop_number)
    #     d_pop_list = split_until_limit(d_pop_list, 0.5, pop_number)

    #     s_pop_list = []
    #     for i in range(len(d_pop_list)):
    #         s_pop_list.append(s_matrix / pop_number)
    #     # ss=np.sum(d_pop_list,axis=0)
    #     # sss=np.sum(d_wave)
    #     f_list = []
    #     u_list = []
    #     for i in range(len(d_pop_list)):
    #         s_time = time.time()
    #         f, u = cls.lp_by_gp_sub(n_matrix, s_pop_list[i], d_pop_list[i], N)
    #         f_list.append(f)
    #         u_list.append(u)
    #         # print(time.time() - s_time)
    #     f = np.sum(f_list, axis=0)
    #     utilization = cls.calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix)
    #     # utilization[mask] = 0

    #     return np.max(utilization), f
    @classmethod
    def lp_by_pop(cls, n_matrix, s_matrix, d_wave, N, pop_number):
        def process_submatrix(args):
            cls, n_matrix, s_sub, d_sub, N = args
            return cls.lp_by_gp_sub(n_matrix, s_sub, d_sub, N)
        import time
        from concurrent.futures import ProcessPoolExecutor

        s_matrix = dict_to_numpy(s_matrix, N)
        d_wave = dict_to_numpy(d_wave, N)

        d_pop_list = create_submatrices(d_wave, pop_number)
        d_pop_list = split_until_limit(d_pop_list, 0.5, pop_number)

        # 分割 s_matrix
        s_pop_list = [s_matrix / pop_number for _ in range(len(d_pop_list))]

        # 准备输入参数
        args_list = [(cls, n_matrix, s_pop_list[i], d_pop_list[i], N) for i in range(len(d_pop_list))]

        # 并行计算
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_submatrix, args_list))

        # 收集结果
        f_list, u_list = zip(*results)
        f = np.sum(f_list, axis=0)
        utilization = cls.calculate_bandwidth_utilization(N, f, d_wave, n_matrix, s_matrix)

        return np.max(utilization), f
    @classmethod
    def lp_top_p(cls, n_matrix, s_matrix, d_wave, N, p, mask_matrix):
        import time
        import numpy as np

        # 将输入矩阵转为 numpy 格式
        s_matrix = dict_to_numpy(s_matrix, N)
        d_wave = dict_to_numpy(d_wave, N)

        # 初始化三维流量矩阵 f
        f_direct = np.zeros((N, N, N))  # 用于保存直连流量
        f_optimized = np.zeros((N, N, N))  # 用于保存优化流量

        # 将需求展平，找出前 p% 的需求
        flat_indices = np.arange(d_wave.size)
        flat_demands = d_wave.flatten()

        top_p_count = int(len(flat_demands) * p / 100)
        top_p_indices = np.argsort(-flat_demands)[:top_p_count]  # 按需求从大到小排序

        # 构造优化需求矩阵 d_top_p
        d_top_p = np.zeros_like(d_wave)
        for idx in top_p_indices:
            i, j = divmod(idx, N)
            d_top_p[i, j] = d_wave[i, j]

        # 构造直连需求矩阵 d_direct
        d_direct = d_wave - d_top_p

        # 优化需求部分
        start_time = time.time()
        # f_partial, u = cls.lp_by_gp_sub(n_matrix, s_matrix, d_top_p, N)  # 假设此函数返回三维矩阵 f_partial
        f_partial, u = cls.lp_by_gp(n_matrix, s_matrix, d_top_p, N, mask_matrix)
        
        # print(f"Optimization Time: {time.time() - start_time}s")

        # 将优化结果赋值给 f_optimized
        f_optimized[:, :, :] = f_partial

        # 处理剩余部分：未优化的需求直接设置为直连流量
        for i in range(N):
            for j in range(N):
                if d_direct[i, j] > 0:
                    f_direct[i, j, j] = 1  # 直连流量写入到 f[i, j, j]

        # 合并优化流量和直连流量
        f_final = f_optimized + f_direct

        # 计算带宽利用率
        utilization = cls.calculate_bandwidth_utilization(N, f_final, d_wave, n_matrix, s_matrix)

        return np.max(utilization), f_final




if __name__ == '__main__':
    # # 初始化
    #
    # pod_count = 5
    # up_link_port_range = [256, 256]
    # up_link_bandwidth_range = [100, 100]
    # traffic_range = [0, 100]
    # error_tolerance = 1e-12
    #
    # pods, traffic_matrix, bandwidth_matrix = init_structure(pod_count, up_link_port_range, up_link_bandwidth_range,
    #                                                         traffic_range, 1)
    #
    # max_bandwidth_direct, direct_matrix, _ = no_by_pass(pod_count, pods, traffic_matrix, bandwidth_matrix,
    #                                                     error_tolerance)
    #
    # R = [pod.R for pod in pods]
    # N = len(pods)
    # u_no_by_pass, n_matrix, path = no_by_pass(N, pods, traffic_matrix, bandwidth_matrix, 0.0001)
    #
    # continue_flag = True
    # u_tmp = float("inf")
    # e = 1e-7
    # utilization_records = []  # 用于记录每次迭代的utilization
    # n_matrix_records = []  # 用于记录每次迭代的n_matrix
    # i = 0
    # R = [pod.R for pod in pods]
    #
    # T_tmp = traffic_matrix
    # f = RouteTool.initialize_f(N, n_matrix)
    #
    # while continue_flag:
    #     u_no_by_pass, n_matrix, path = no_by_pass(N, pods, T_tmp, bandwidth_matrix, e)
    #     utilization = RouteTool.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
    #     # utilization = np.clip(utilization, a_min=u_no_by_pass, a_max=u_no_by_pass)
    #     utilization_records.append(utilization.copy())  # 记录当前迭代的utilization
    #     R_c = calculate_R_c(n_matrix, pods, N)
    #     RouteTool.update_n_matrix_based_on_utilization(N, R, R_c, n_matrix, utilization)
    #     start_time = time.time()
    #     f, u_now = RouteTool.lp_rapid(n_matrix, bandwidth_matrix, traffic_matrix, N, 200, tol=1e-6)
    #     print(f"time:{time.time() - start_time},result:{u_now}")
    #     start_time = time.time()
    #     f1, u_now1 = RouteTool.lp_by_gp(n_matrix, bandwidth_matrix, traffic_matrix, N)
    #     f1 = dict_to_numpy(f1, N)
    #     print(f"time:{time.time() - start_time},result:{u_now1}")
    #     traffic_matrix = dict_to_numpy(traffic_matrix, N)
    #
    #     utilization = RouteTool.calculate_bandwidth_utilization(N, f, traffic_matrix, n_matrix, bandwidth_matrix)
    #
    #     utilization_records.append(utilization.copy())  # 记录当前迭代的utilization
    #     if u_tmp - u_now < 1e-7:
    #         continue_flag = False
    #         print(f"正在最后一次即第{i}次迭代,当前最大带宽利用率为{u_tmp}")
    #     else:
    #         T_tmp = RouteTool.get_traffic(N, f, traffic_matrix)
    #         u_tmp = u_now
    #         print(f"正在第{i}次迭代,当前最大带宽利用率为{u_tmp}")
    #         i += 1
    #
    #     # utilization_records.append(utilization.copy())  # 记录当前迭代的utilization
    #     n_matrix_records.append(n_matrix.copy())  # 记录当前迭代的n_matrix
    #
    # f, max_value = RouteTool.lp_rapid(n_matrix, bandwidth_matrix, traffic_matrix, N, 50)
    pass
