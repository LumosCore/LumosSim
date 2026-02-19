
import numpy as np
import time
import networkx as nx
def calculate_R_c(n_matrix, R, N):
    """
    计算每个节点的实际连接数R_c。

    Args:
        n_matrix (Union[Dict[Tuple[int, int], int], np.ndarray]): 拓扑矩阵，可以是字典或NumPy数组形式。
        pods (List[Pod]): Pod对象的列表，其中每个对象都有R属性表示其最大连接数。
        N (int): 节点总数。

    Returns:
        Tuple[List[int], Union[int, str]]: 如果计算成功，返回一个元组，其中第一个元素是包含每个节点实际连接数的列表，第二个元素为None；
                                            如果计算失败（即存在某个节点的实际连接数大于其最大连接数），返回元组(0, "fail")。

    Raises:
        ValueError: 如果n_matrix既不是字典也不是NumPy数组，将引发此异常。

    """
    R_c = []
    for i in range(N):
        if isinstance(n_matrix, dict):
            # 如果n_matrix是字典形式的矩阵，使用字典访问和求和
            tmp = sum(n_matrix.get((i, j), 0) for j in range(N))
        elif isinstance(n_matrix, np.ndarray):
            # 如果n_matrix是NumPy数组，使用NumPy的sum函数
            tmp = np.sum(n_matrix[i, :])
        else:
            # import pdb;pdb.set_trace()
            raise ValueError("Unsupported matrix type")

        if tmp > R[i]:
            return 0, "fail"
        R_c.append(tmp)
    return R_c
def get_R_from_pods(pods):
    return [pod.R for pod in pods]
def get_3d_array(allocations):
        """
        获取当前累积的所有 (P, P) 数组组成的 (P, P, T) 三维数组。
        
        返回:
            numpy.ndarray, 形状为 (P, P, T)
        """
        
        T = len(allocations)
        result = np.stack(allocations, axis=2)
        return result
def compute_routing(N, d_wave, R_src, R_dst, s_matrix, e=0.01):
    """
    考虑Pod端口限制的分阶段最大加权匹配路由算法

    Args:
        N (int): Pod数量
        d_wave (np.ndarray): 初始流量需求矩阵，shape=(N, N)
        R_src (np.ndarray): 每个Pod的发送端口数，shape=(N,)
        R_dst (np.ndarray): 每个Pod的接收端口数，shape=(N,)
        s_matrix (np.ndarray): 超链路容量矩阵，shape=(N, N)
        e (float): 精度阈值，需求小于该值时视为0

    Returns:
        Tuple[np.ndarray, float]: 路由分配矩阵和总吞吐量
    """
    # 初始化路由矩阵、剩余需求和端口状态
    
        # 将 d_wave 从字典转换为 numpy 数组
    if isinstance(d_wave, dict):
        remaining_demand = np.zeros((N, N))
        for (i, j), value in d_wave.items():
            remaining_demand[i, j] = value
    else:
        remaining_demand = np.copy(d_wave)
    if isinstance(s_matrix, dict):
        s_matrix_array = np.zeros((N, N))
        for (i, j), value in s_matrix.items():
            s_matrix_array[i, j] = value
        s_matrix = s_matrix_array
    src_ports = np.copy(R_src).astype(int)   # 剩余发送端口数
    dst_ports = np.copy(R_dst).astype(int)   # 剩余接收端口数
    optimal_u = 0.0
    n_matrix = np.zeros_like(remaining_demand, dtype=np.float64)
    # 分阶段处理，最多进行max(R_src, R_dst)轮
    allocations = []
    max_rounds = max(np.max(R_src), np.max(R_dst))
    for _ in range(max_rounds):
        # 动态构建二分图：仅包含有剩余端口的Pod
        G = nx.Graph()
        source_nodes = [i for i in range(N) if src_ports[i] > 0]
        dest_nodes = [N + j for j in range(N) if dst_ports[j] > 0]
        G.add_nodes_from(source_nodes, bipartite=0)
        G.add_nodes_from(dest_nodes, bipartite=1)

        # 添加有效边（排除无端口或低需求的链路）
        has_valid_edges = False
        for i in source_nodes:
            for j in [node for node in dest_nodes if dst_ports[node - N] > 0]:
                alloc = min(remaining_demand[i, j - N], s_matrix[i, j - N])
                if alloc >= e:
                    G.add_edge(i, j, weight=alloc)
                    has_valid_edges = True

        if not has_valid_edges:
            break  # 提前终止：无有效链路

        # 计算最大权重匹配
        matching = nx.max_weight_matching(G, maxcardinality=False)

        # 更新路由和端口状态
        current_alloc = np.zeros_like(n_matrix)
        for u, v in matching:
            if u < N:  # 确保u是源节点，v是目标节点
                src, dest = u, v - N
            else:
                src, dest = v, u - N
            if 0 <= src < N and 0 <= dest < N:
                # 计算实际分配值
                alloc = min(remaining_demand[src, dest], s_matrix[src, dest])
                current_alloc[src, dest] = alloc
                # 扣除端口（确保不越界）
                src_ports[src] = max(src_ports[src] - 1, 0)
                dst_ports[dest] = max(dst_ports[dest] - 1, 0)
                # 更新剩余需求
                remaining_demand[src, dest] = max(remaining_demand[src, dest] - alloc, 0)
            n_matrix[src,dest] += 1
        # print("debug current_alloc")
        # print(current_alloc)
        # 累加结果
        # n_matrix += current_alloc
        allocations.append(current_alloc)
        optimal_u += current_alloc.sum()

        # 检查是否所有需求已满足
        if np.all(remaining_demand <= e):
            break
        #   step1
    R_c = calculate_R_c(n_matrix, R_src, N)
    #   step2
    # print("step2")
    U_matrix = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                U_matrix[(i, j)] = 0
            else:
                if n_matrix[(i, j)] == 0:
                    U_matrix[(i, j)] = 0
                else:
                    U_matrix[(i, j)] = d_wave[(i, j)] / (min(n_matrix[(i, j)],n_matrix[(i, j)]) * s_matrix[(i, j)])
    u_sort_list = sorted(U_matrix.keys(), key=lambda k: U_matrix[k], reverse=True)
    u_max_1 = U_matrix[u_sort_list[0]]
    x_ijk = get_3d_array(allocations)
    return n_matrix, u_max_1, x_ijk

def bg(r, d_wave, base_capacity):
    # 设置基础参数
    peak_demand = d_wave
    s_matrix = np.ones(d_wave.shape)*base_capacity
    d_wave = d_wave.reshape(s_matrix.shape)

    N = s_matrix.shape[0]
    R = [r]*N
 
    T_tmp = d_wave

    u_tmp = float("inf")
    i = 1
    utilization_records = []  # 用于记录每次迭代的utilization
    n_matrix_records = []  # 用于记录每次迭代的n_matrix
    n_matrix, u_tmp, x_ijt = compute_routing(N, T_tmp, R, R, s_matrix, 1e-7)
    return x_ijt

if __name__ == '__main__':
    d_wave = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    base_capacity = 1
    r = 2
    x_ijt = bg(r, d_wave, base_capacity)
    # print(x_ijt)
    # print(x_ijt.shape)
    # for oxc_id in range(r):
    #     print(x_ijt[:,:,oxc_id])
