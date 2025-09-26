import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


# def plot_data(*arrays):
#     """
#     绘制给定数组的数据图像。

#     Args:
#         *arrays: 一个或多个一维或二维的numpy数组。

#     Returns:
#         None.

#     Raises:
#         ValueError: 如果输入数组不是二维数组，或者输入的数据格式不支持时，将引发此异常。

#     """
#     if len(arrays) == 1:
#         array = arrays[0]
#         if array.ndim == 2:
#             # 如果输入是二维数组，则使用seaborn绘制更美观的热力图
#             plt.figure(figsize=(8, 6))
#             sns.heatmap(array, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True, square=True, linewidths=.5,
#                         linecolor='white')
#             plt.title('Heatmap')
#         else:
#             raise ValueError("输入的数组不是二维数组")
#     elif all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in arrays):
#         plt.figure(figsize=(10, 6))

#         for idx, arr in enumerate(arrays):
#             plt.plot(arr, label=f'Array {idx + 1}')

#         plt.title('Multiple Arrays Plot')
#         plt.xlabel('Index')
#         plt.ylabel('Value')
#         plt.legend()
#         plt.grid(True)

#     else:
#         raise ValueError("输入的数据格式不支持")

#     plt.show()


def convert_to_dict(matrix):
    """
    将numpy矩阵转换为字典形式，键为索引元组，值为对应元素值。

    Args:
        matrix (np.ndarray): 需要转换的numpy矩阵。

    Returns:
        Union[Dict[Tuple[int, int], float], Any]: 转换后的字典形式，如果输入不是numpy矩阵，则原样返回。

    """
    # 判断输入是否是numpy矩阵
    if isinstance(matrix, np.ndarray):
        # 初始化一个空字典
        matrix_dict = {}
        # 遍历矩阵的每一个元素
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # 将索引元组和对应的值存入字典中
                matrix_dict[(i, j)] = matrix[i, j]
        return matrix_dict
    else:
        # 如果不是numpy矩阵，原样返回
        return matrix


def dict_to_numpy(matrix_dict, N):
    """
    将字典格式的矩阵转换为numpy格式的矩阵

    Args:
        matrix_dict (dict or np.ndarray): 字典格式的矩阵或numpy格式的矩阵
        N (int): 矩阵的维度

    Returns:
        np.ndarray: 转换后的numpy格式的矩阵

    """
    if isinstance(matrix_dict, np.ndarray):
        numpy_matrix = matrix_dict
    else:
        dim = len(next(iter(matrix_dict.keys())))
        if dim != 2:
            numpy_matrix = np.zeros((N, N, N))
            # 遍历字典并填充矩阵
            for key, value in matrix_dict.items():
                i, j, k = key  # 假设字典的键是一个元组(i, j)
                numpy_matrix[i, j, k] = value
        else:
            # 创建N x N的全零矩阵
            numpy_matrix = np.zeros((N, N))
            # 遍历字典并填充矩阵
            for key, value in matrix_dict.items():
                i, j = key  # 假设字典的键是一个元组(i, j)
                numpy_matrix[i, j] = value

    return numpy_matrix


def create_submatrices(original_matrix, n):
    original_matrix = np.array(original_matrix)

    # 获取原矩阵中非零元素的索引
    non_zero_indices = np.argwhere(original_matrix != 0)

    # 如果非零元素的数量少于 n，抛出异常
    if len(non_zero_indices) < n:
        raise ValueError("Not enough non-zero elements to create the required submatrices.")

    # 随机打乱非零元素的索引
    np.random.shuffle(non_zero_indices)

    # 将索引分成 n 组
    groups = np.array_split(non_zero_indices, n)

    # 初始化子矩阵列表
    submatrices = []
    shape = original_matrix.shape

    # 为每个子矩阵分配一组元素
    for group in groups:
        submatrix = np.zeros(shape, dtype=original_matrix.dtype)
        for row, col in group:
            submatrix[row, col] = original_matrix[row, col]
        submatrices.append(submatrix)

    return submatrices


# Sorting by the "traffic", here represented by the sum of non-zero elements in each submatrix
def sort_submatrices_by_traffic(submatrices):
    return sorted(submatrices, key=lambda x: np.sum(x), reverse=True)


# Split the largest submatrix based on the algorithm
def split_largest_matrix(submatrices, split_ratio=0.5):
    largest_matrix = submatrices.pop(0)  # Pop the largest matrix (already sorted)
    non_zero_indices = np.argwhere(largest_matrix != 0)

    if len(non_zero_indices) <= 1:
        # If there is only one non-zero element or none, we can't split further
        return submatrices

    # Randomly shuffle the indices to split
    np.random.shuffle(non_zero_indices)

    # Split into two halves
    half = int(len(non_zero_indices) * split_ratio)
    group1, group2 = non_zero_indices[:half], non_zero_indices[half:]

    # Create two new matrices
    shape = largest_matrix.shape
    submatrix1 = np.zeros(shape, dtype=largest_matrix.dtype)
    submatrix2 = np.zeros(shape, dtype=largest_matrix.dtype)

    for row, col in group1:
        submatrix1[row, col] = largest_matrix[row, col]

    for row, col in group2:
        submatrix2[row, col] = largest_matrix[row, col]

    # Add the two new matrices back to the list
    submatrices.extend([submatrix1, submatrix2])

    return submatrices


def split_until_limit(submatrices, t, n):
    # Initialize a queue (simulating a max heap based on traffic/size)
    queue = sort_submatrices_by_traffic(submatrices)  # Assumes sorted initially by traffic (size)

    # While queue length is less than or equal to (1 + t) * n, continue splitting
    while len(queue) <= (1 + t) * n:
        # Split the largest matrix in the queue
        queue = split_largest_matrix(queue)

    return queue



import numpy as np
import os
import glob
from collections import defaultdict
import sys

import scipy
# from sklearn.cluster import KMeans

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import DATA_DIR


def Get_peak_demand(dm_list):
    """Get the peak demand from the demand matrix."""
    dm_matrixs = np.array(dm_list)
    predict_dm = np.max(dm_matrixs, axis=0)
    return predict_dm


def Get_edge_to_path(topology, candidate_path):
    """ Get the mapping from edge to path."""
    if candidate_path == None:
        return None
    edge_to_path = {}
    for edge in topology.edges:
        edge_to_path[(int(edge[0]), int(edge[1]))] = []
    for src in topology.nodes:
        for dst in topology.nodes:
            if src != dst:
                for index, path in enumerate(candidate_path[(src, dst)]):
                    for i in range(len(path) - 1):
                        edge_to_path[(int(path[i]), int(path[i + 1]))].append((int(src), int(dst), index))
    return edge_to_path


def linear_get_dir(props, is_test):
    """Get the train or test directory for the given topology."""
    postfix = "test" if is_test else "train"
    return os.path.join(DATA_DIR, props.topo_name, postfix)


def linear_get_hists_from_folder(folder):
    """Get the list of histogram files from the given folder."""
    hists = sorted(glob.glob(folder + "/*.hist"))
    return hists


def paths_from_file(paths_file, num_nodes):
    """Get the paths from the file."""
    pij = defaultdict(list)
    pid = 0
    with open(paths_file, 'r') as f:
        lines = sorted(f.readlines())
        lines_dict = {line.split(":")[0]: line for line in lines if line.strip() != ""}
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                try:
                    if "%d %d" % (src, dst) in lines_dict:
                        line = lines_dict["%d %d" % (src, dst)].strip()
                    else:
                        line = [l for l in lines if l.startswith("%d %d:" % (src, dst))]
                        if line == []:
                            continue
                        line = line[0]
                        line = line.strip()
                    if not line: continue
                    i, j = list(map(int, line.split(":")[0].split(" ")))
                    paths = line.split(":")[1].split(",")
                    for p_ in paths:
                        node_list = list(map(int, p_.split("-")))
                        pij[(i, j)].append(node_list)
                        pid += 1
                except Exception as e:
                    print(e)
                    import pdb;
                    pdb.set_trace()
    return pij


def Get_common_cases_tms(hist_tms):
    """Get the common cases traffic demands from the history traffic demands.

    Args:
        hist_tms: the history traffic demands.
    """
    # Computing the convex hull can be challenging when the length of hist_tms is particularly large,
    # or when the dimensionality of each tm is very high.
    # Therefore, we use all hist_tms as common_cases_tms.

    # hull = ConvexHull(hist_tms)
    # hull_vertices = hull.vertices
    # common_case_tms = [hist_tms[i] for i in hull_vertices]
    common_case_tms = hist_tms
    return common_case_tms


def restore_flattened_to_original(flattened_f, original_shape, mask_3d):
    # 创建一个与原始 f 形状相同的空数组，用于恢复数据
    restored_f = np.zeros(original_shape)

    # 将展平后的数据根据掩码填回原数组
    if isinstance(flattened_f,np.matrix):
        restored_f[mask_3d]=flattened_f.A.flatten()
    elif isinstance(flattened_f,scipy.sparse._coo.coo_matrix):
        restored_f[mask_3d] = flattened_f.toarray().flatten()
    else:
        restored_f[mask_3d] = flattened_f.flatten()


    return restored_f


def dict_to_numpy_3d(path_weights):
    """
    将包含键格式为 'w_i_j_k' 的字典转换为 3D NumPy 数组，未被赋值的位置用 0 填充。

    Args:
        path_weights (dict): 键格式为 'w_i_j_k' 的字典，值为浮点数。

    Returns:
        np.ndarray: 一个 3D NumPy 数组，未被赋值的位置用 0 填充。
    """
    # 提取最大 i, j, k 的索引值，以便初始化数组
    max_i = max(int(key.split('_')[1]) for key in path_weights.keys()) + 1
    max_j = max(int(key.split('_')[2]) for key in path_weights.keys()) + 1
    max_k = max(int(key.split('_')[3]) for key in path_weights.keys()) + 1

    # 初始化三维数组，使用 0 填充
    array = np.zeros((max_i, max_j, max_k))

    # 填充数组
    for key, value in path_weights.items():
        _, i, j, k = key.split('_')
        i, j, k = int(i), int(j), int(k)
        array[i, j, k] = value

    return array


def numpy_3d_to_dict(array, topology, candidate_path):
    """
    将 3D NumPy 数组按照指定的 'w_i_j_k' 键格式转换为字典。
    仅存储非零值的元素。

    Args:
        array (np.ndarray): 3D NumPy 数组。
        topology: 包含节点数量的拓扑结构对象。
        candidate_path (dict): 包含每对节点 (i, j) 的候选路径集合。

    Returns:
        dict: 键为 'w_i_j_k' 的字典，值为浮点数，仅包含非零元素。
    """
    path_weights = {}

    # 遍历节点对和路径编号
    for i in range(topology.number_of_nodes()):
        for j in range(topology.number_of_nodes()):
            if j != i:
                num_paths = len(candidate_path[(i, j)])  # 获取从 i 到 j 的路径数
                for k in range(num_paths):
                    # if array[i, j, k] != 0:  # 只存储非零元素
                    key = f'w_{i}_{j}_{k}'
                    path_weights[key] = array[i, j, k]

    return path_weights

def dict_to_numpy(matrix_dict, N):
    """
    将字典格式的矩阵转换为numpy格式的矩阵

    Args:
        matrix_dict (dict or np.ndarray): 字典格式的矩阵或numpy格式的矩阵
        N (int): 矩阵的维度

    Returns:
        np.ndarray: 转换后的numpy格式的矩阵

    """
    if isinstance(matrix_dict, np.ndarray):
        numpy_matrix = matrix_dict
    else:
        dim = len(next(iter(matrix_dict.keys())))
        if dim != 2:
            numpy_matrix = np.zeros((N, N, N))
            # 遍历字典并填充矩阵
            for key, value in matrix_dict.items():
                i, j, k = key  # 假设字典的键是一个元组(i, j)
                numpy_matrix[i, j, k] = value
        else:
            # 创建N x N的全零矩阵
            numpy_matrix = np.zeros((N, N))
            # 遍历字典并填充矩阵
            for key, value in matrix_dict.items():
                i, j = key  # 假设字典的键是一个元组(i, j)
                numpy_matrix[i, j] = value

    return numpy_matrix

def create_submatrices(original_matrix, n):
    original_matrix = np.array(original_matrix)

    # 获取原矩阵中非零元素的索引
    non_zero_indices = np.argwhere(original_matrix != 0)

    # 如果非零元素的数量少于 n，抛出异常
    if len(non_zero_indices) < n:
        raise ValueError("Not enough non-zero elements to create the required submatrices.")

    # 随机打乱非零元素的索引
    np.random.shuffle(non_zero_indices)

    # 将索引分成 n 组
    groups = np.array_split(non_zero_indices, n)

    # 初始化子矩阵列表
    submatrices = []
    shape = original_matrix.shape

    # 为每个子矩阵分配一组元素
    for group in groups:
        submatrix = np.zeros(shape, dtype=original_matrix.dtype)
        for row, col in group:
            submatrix[row, col] = original_matrix[row, col]
        submatrices.append(submatrix)

    return submatrices


# Sorting by the "traffic", here represented by the sum of non-zero elements in each submatrix
def sort_submatrices_by_traffic(submatrices):
    return sorted(submatrices, key=lambda x: np.sum(x), reverse=True)


# Split the largest submatrix based on the algorithm
def split_largest_matrix(submatrices, split_ratio=0.5):
    largest_matrix = submatrices.pop(0)  # Pop the largest matrix (already sorted)
    non_zero_indices = np.argwhere(largest_matrix != 0)

    if len(non_zero_indices) <= 1:
        # If there is only one non-zero element or none, we can't split further
        return submatrices

    # Randomly shuffle the indices to split
    np.random.shuffle(non_zero_indices)

    # Split into two halves
    half = int(len(non_zero_indices) * split_ratio)
    group1, group2 = non_zero_indices[:half], non_zero_indices[half:]

    # Create two new matrices
    shape = largest_matrix.shape
    submatrix1 = np.zeros(shape, dtype=largest_matrix.dtype)
    submatrix2 = np.zeros(shape, dtype=largest_matrix.dtype)

    for row, col in group1:
        submatrix1[row, col] = largest_matrix[row, col]

    for row, col in group2:
        submatrix2[row, col] = largest_matrix[row, col]

    # Add the two new matrices back to the list
    submatrices.extend([submatrix1, submatrix2])

    return submatrices


def split_until_limit(submatrices, t, n):
    # Initialize a queue (simulating a max heap based on traffic/size)
    queue = sort_submatrices_by_traffic(submatrices)  # Assumes sorted initially by traffic (size)

    # While queue length is less than or equal to (1 + t) * n, continue splitting
    while len(queue) <= (1 + t) * n:
        # Split the largest matrix in the queue
        queue = split_largest_matrix(queue)

    return queue
