import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.colors import LogNorm

def plot_traffic_matrix(traffic_matrix, title, vmax, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.5)
    sns.heatmap(traffic_matrix, cmap='YlGnBu', annot=False, cbar=True, vmin=0.1, vmax=vmax, norm=LogNorm()) #, norm=LogNorm()
    plt.title(title)
    plt.xlabel('Destination')
    plt.ylabel('Source')
    if save_path:
        plt.savefig(f'{save_path}/{title}.png')
    # plt.show()
    plt.close()


def read_traffic_matrix_log(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            time, traffic_matrix_str = line.strip().split(',')
            traffic_matrix = np.array(list(map(float, traffic_matrix_str.split(' '))))
            shape = int(np.sqrt(traffic_matrix.shape[0]))
            traffic_matrix = traffic_matrix.reshape((shape, shape))
            # traffic_matrix[traffic_matrix > 10000000] = 0
            yield time, traffic_matrix


def main():
    file_path = 'xxx/rapidAIsim-moe/large_exp_4096GPU_beta_analysis_20250131/beta_500/ele/traffic_matrix.log'
    traffic_matrices = []
    time_interval = 300
    curr_matrix = None
    for time, traffic_matrix in read_traffic_matrix_log(file_path):
        if curr_matrix is None or int(time) < 100700:
            curr_matrix = traffic_matrix
            continue
        if int(time) % time_interval == 0:
            traffic_matrices.append(curr_matrix)
            curr_matrix = None
            continue
        curr_matrix += traffic_matrix
    # traffic_matrices = np.array(traffic_matrices)
    vmax = max([np.max(matrix) for matrix in traffic_matrices])
    print(vmax)
    # time_interval //= 1000
    old_matrix = np.zeros_like(traffic_matrix)
    cos_list = []
    for i, traffic_matrix in tqdm.tqdm(enumerate(traffic_matrices)):
        
        flat_a = old_matrix.flatten()
        flat_b = traffic_matrix.flatten()
        cosine_similarity = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
        cos_list.append(cosine_similarity)
        # plot_traffic_matrix(traffic_matrix, f'Traffic Matrix at {(i + 1) * time_interval}s', vmax,
        #                     save_path='xxx/rapidAIsim-moe/large_exp_4096GPU_beta_analysis_20250131/base_conf/traffic_matrix_fig')
        old_matrix = traffic_matrix
    # print(traffic_matrices.shape)
    old_matrix = old_matrix[1:]

    # 如果你知道每个cos值对应的x轴位置或时间点，请提供该信息。
    # 这里我们简单地使用索引作为x轴的位置
    x = np.arange(len(cos_list))*time_interval

    # 创建图形
    plt.figure(figsize=(6, 4))  # 可以根据需要调整图形大小

    # 绘制折线图
    plt.plot(x, cos_list, marker='o')  # 使用圆圈标记点

    # 添加标题和坐标轴标签
    plt.title('Cosine Similarity Change of Traffic Matrix')
    plt.xlabel('Time(s)')
    plt.ylabel('Cosine Similarity')

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # 如果你知道每个点的具体含义（例如时间点），可以修改x轴的标签
    # plt.xticks(x, ['Point1', 'Point2', 'Point3', ...])

    # 显示网格
    plt.grid(True)
    save_path = 'xxx/rapidAIsim-moe/large_exp_4096GPU_beta_analysis_20250131/base_conf/traffic_matrix_fig'
    plt.savefig(f'{save_path}/cos_index.png')


if __name__ == '__main__':
    main()
