from rapidAIsim.scheduler.ocsexpander.cijt_solver import CijtSolver
import numpy as np
import time

def tiny_scale_test():
    c_ij = np.zeros((4, 4), dtype=np.int64)
    c_ij[0,1] = 8
    c_ij[0,2] = 8
    c_ij[1,0] = 8
    c_ij[1,3] = 8
    c_ij[2,0] = 8
    c_ij[2,3] = 8
    c_ij[3,1] = 8
    c_ij[3,2] = 7
    print(c_ij)
    solver = CijtSolver(c_ij, 16, 4, 16)
    solver.solve()
    c_ij.tofile('c_ij.csv', ',')

# def test_cijt_solver():
#     size = 4096
#     np.random.seed(5)
#     c_ij = np.zeros((size, size), dtype=np.int32)
#     selected_index = np.random.choice(np.arange(size * size), 32)
#     selected_num = np.random.randint(0, 8, size)
#     # print(selected_index)
#     for i, num in zip(selected_index, selected_num):
#         line = i // size
#         column = i % size
#         c_ij[line, column] = num
#     for i in range(size):
#         c_ij[i, i] = 0
#     c_ij = c_ij + c_ij.T
#     # print(c_ij)
#     solver = CijtSolver(c_ij, 16, size, 16)
#     solver.solve()
#     c_ij.tofile('c_ij.csv', ',')
    
def test_cijt_solver(exp_size):
    size = exp_size
    np.random.seed(5)
    c_ij = np.zeros((size, size), dtype=np.int32)
    selected_index = np.random.choice(np.arange(size * size), 32)
    selected_num = np.random.randint(0, 8, size)
    # print(selected_index)
    for i, num in zip(selected_index, selected_num):
        line = i // size
        column = i % size
        c_ij[line, column] = num
    for i in range(size):
        c_ij[i, i] = 0
    c_ij = c_ij + c_ij.T
    # print(c_ij)
    solver = CijtSolver(c_ij, 16, size, 16)
    solver.solve()
    c_ij.tofile('c_ij.csv', ',')


if __name__ == '__main__':
    start_time = time.time()
    test_cijt_solver(128)
    print("time cost:" ,time.time() - start_time)
    start_time = time.time()
    test_cijt_solver(256)
    print("time cost:" ,time.time() - start_time)
    start_time = time.time()
    test_cijt_solver(512)
    print("time cost:" ,time.time() - start_time)
    start_time = time.time()
    test_cijt_solver(1024)
    print("time cost:" ,time.time() - start_time)
    # tiny_scale_test()
