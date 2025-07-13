import random
import numpy as np
from collections import deque

class Scheduler:
    '''
        generate vertex batches for each parallel iteration
    '''
    def __init__(self, initial_partition):
        # shuffle vertices in each partition
        self.parts = initial_partition.copy()
        for i in range(len(self.parts)):
            #random.shuffle(self.parts[i])
            self.parts[i] = deque(self.parts[i])
        
        self.pid_start = 0
        self.part_num = len(self.parts)
        self.batch = {"v": [], "src":[], "size": 0}

    def getVertexBatch(self, size_required):
        '''
            generate vertex batches for each parallel iteration
        '''
        # clear batch information
        self.batch = {"v": [], "src":[], "size": 0}
        actual_size = 0
        while actual_size < size_required:
            has_vertex = False
            for _ in range(self.part_num):
                self.pid_start = (self.pid_start + 1) % self.part_num
                pid = self.pid_start # start extracting vertices from this partition
                if len(self.parts[pid]) == 0:
                    continue

                has_vertex = True
                v = self.parts[pid].popleft()
                self.batch["v"].append(v)
                self.batch["src"].append(pid)
                actual_size += 1
                if actual_size >= size_required:
                    break
            if not has_vertex: # no more vertices in all partitions
                break

        self.batch["size"] = actual_size
        return self.batch, has_vertex
