import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait


def transform_log_to_hist_stage(traffic_file, start_hist_file, single_hist_len,
                                start_line_num, max_end_line_num, dir, offset):
    if start_line_num >= max_end_line_num:
        return
    read_file_ptr = open(traffic_file, mode='r')
    read_file_ptr.seek(offset, 0)
    for _ in range(start_line_num):
        next(read_file_ptr)

    demands = []
    for _ in range(min(single_hist_len, max_end_line_num - start_line_num)):
        line = read_file_ptr.readline()
        demand = line.split(',')[1]
        demand = map(float, demand.split(' '))
        demands.append(list(demand))
    hist = np.array(demands)
    shape = np.array(hist.shape)
    np.savez_compressed(f'{dir}/{start_hist_file}.npz', arr_0=hist, shape=shape)


def transform_log_to_hist_multiprocess(traffic_matrix_file: str, file_length: int,
                                       start_hist_file: int, num_processers: int, single_hist_len: int,
                                       skip_len: int):
    """
    多线程地将traffic_matrix_file转换为hist文件。适用于traffic_matrix_file较大的情况。

    Args:
        traffic_matrix_file (str): 流量矩阵文件路径
        file_length (int): 流量矩阵文件的行数
        start_hist_file (int): 开始hist文件的编号
        num_processers (int): 并行度
        single_hist_len (int): 保存的单个hist文件的长度
        skip_len (int): 跳过的首尾行数
    """
    read_file_ptr = open(traffic_matrix_file, mode='r')
    # 跳过首尾
    for _ in range(skip_len):
        read_file_ptr.readline()
    file_length -= 2 * skip_len
    offset = read_file_ptr.tell()
    # 区分train和test
    train_length = int(file_length * 0.833)
    # train_length = 0
    stage_length = num_processers * single_hist_len
    stage_num = 0
    while stage_num * stage_length < train_length:
        params = [(traffic_matrix_file, start_hist_file + i, single_hist_len,
                   stage_num * stage_length + i * single_hist_len, train_length, 'train', offset)
                  for i in range(num_processers)]
        with ProcessPoolExecutor(num_processers) as executor:
            futures = [executor.submit(transform_log_to_hist_stage, *param) for param in params]
            wait(futures)
        stage_num += 1
        start_hist_file += num_processers
    # 计算test开始时文件的偏移量
    for _ in range(train_length):
        read_file_ptr.readline()
    offset = read_file_ptr.tell()
    file_length -= train_length
    start_hist_file -= num_processers * stage_num
    stage_num = 0
    while stage_num * stage_length < file_length:
        params = [(traffic_matrix_file, start_hist_file + i, single_hist_len,
                   stage_num * stage_length + i * single_hist_len, file_length, 'test', offset)
                  for i in range(num_processers)]
        with ProcessPoolExecutor(num_processers) as executor:
            futures = [executor.submit(transform_log_to_hist_stage, *param) for param in params]
            wait(futures)
        stage_num += 1
        start_hist_file += num_processers


if __name__ == '__main__':
    transform_log_to_hist_multiprocess(
        'traffic_matrix_b700.log', 7117, 1, 4, 2000, 200
    )
