import numpy as np

'''
class switch:
    def __init__(self, port_num=64):
        self.port_num = port_num
        self.free_port_num = self.port_num
        self.port_status = np.zeros(self.port_num)

    def take_up_port_by_index(self, index):
        assert  self.port_status[index] == 0
        #self.port_status[index] = 1
        self.free_port_num -= 1#= self.port_num - np.sum(self.port_status)

    def free_port_index(self, index):
        #self.port_status[index] = 0
        self.free_port_num += 1#self.port_num - np.sum(self.port_status)

    def take_up_port_by_num(self, num):
        print("num:", num)
        print("free port num:", self.free_port_num)
        assert  num <= self.free_port_num
        self.free_port_num -= num


    def free_port_by_num(self, num):
        print("num:", num)
        print("free port num:", self.free_port_num)
        assert  num <= (self.port_num - self.free_port_num)
        self.free_port_num += num


    def ports_free(self, port_set):
        if self.free_port_num >= len(port_set): return True
        else: return False
'''
class switch:
    def __init__(self, port_num=64):
        self.port_num = port_num
        self.free_port_num = self.port_num
        self.port_status = np.zeros(self.port_num)

    def take_up_port_by_index(self, index):
        assert  self.port_status[index] == 0
        self.port_status[index] = 1
        self.free_port_num = self.port_num - np.sum(self.port_status)

    def free_port_index(self, index):
        assert self.port_status[index] == 1
        self.port_status[index] = 0
        self.free_port_num = self.port_num - np.sum(self.port_status)

    def take_up_port_by_num(self, num):
        #print("num:", num)
        #print("free port num:", self.free_port_num)
        assert  num <= self.free_port_num
        count = 0
        for index, status in enumerate(self.port_status):
            if status == 0:
                self.port_status[index] = 1
                count += 1
            if count >= num:
                self.free_port_num = self.port_num - np.sum(self.port_status)
                return

    def free_port_by_num(self, num):
        #print("num:", num)
        #print("free port num:", self.free_port_num)
        assert  num <= (self.port_num - self.free_port_num)
        count = 0
        for index, status in enumerate(self.port_status):
            if status == 1:
                self.port_status[index] = 0
                count += 1
            if count >= num:
                self.free_port_num = self.port_num - np.sum(self.port_status)
                return


    def ports_free(self, port_set):
        for p in port_set:
            if self.port_status[p] == 1: return False
        return True