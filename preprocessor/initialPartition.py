from collections import defaultdict, deque
import numpy as np

'''
    generate balanced initial partition based on 
'''
class InitialPartition:
    def __init__(self, nz_vertices, w_vertices,
                 w_part_max, k):
        self.part = defaultdict(list)
        self.w_part = np.zeros(k, dtype=np.int64)
        self.ignored = deque()
        self.balancedPartition(nz_vertices, w_vertices, w_part_max, k)

    def balancedPartition(self, nz_vertices, w_vertices, 
                          w_part_max, k):
        '''
            sequentially assign vertices to parts that fulfill the balance constraint
        '''
        pid = 0
        for i in range(len(nz_vertices)):
            v = nz_vertices[i]
            w_v = w_vertices[v]
            attempts = 0
            while self.w_part[pid] + w_v > w_part_max:
                pid = (pid + 1) % k
                attempts += 1
                if attempts >= k:
                    #print(f"vertex {v} found no suitable part, {len(nz_vertices)-i} remaining")
                    self.ignored.append(v)
                    break
            if attempts >= k:
                continue
            self.part[pid].append(v)
            self.w_part[pid] += w_v
            
    def getInitialPartition(self):
        return self.part
    
    def getInitialPartWeight(self):
        return self.w_part
    
    def getIgnoredVertices(self):
        return self.ignored
    
    def saveToFile(self, path):
        with open(path, 'w') as f:
            for i in range(len(self.part)):
                f.write(f'{i}:{",".join(map(str, self.part[i]))}\n')


