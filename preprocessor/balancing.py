from collections import defaultdict, deque
import numpy as np

class maxBalanceSubmoveFinder:
    def __init__(self, vertices, src, dst, part_weight, L_max, k):
        '''
            vertices, src, dst are at the same length--number of vertices computed simultaneously
        '''
        self.moves = {"v": [], "src": [], "dst": []}
        self.moves["v"] = vertices
        self.moves["src"] = src
        self.moves["dst"] = dst
        self.map_dst2move = defaultdict(list)
        for i in range(len(vertices)-1, -1, -1):
            self.map_dst2move[dst[i]].append(i)

        self.impacted_parts = set() # all parts that are impacted by the moves
        self.imbalance_parts = set()
        self.was_imbalance = set()
        self.is_imbalance_dst = np.zeros(k, dtype=bool)
        self.exception = set() # inherently imbalanced parts

        self.cause = defaultdict(deque)
        self.has_found_cause = np.zeros(k, dtype=bool)

        self.normal_moves = set(range(len(vertices)))
        self.vulnerable_moves = deque()
        self.jettison_moves = deque()

        for i in range(len(vertices)):
            self.impacted_parts.add(dst[i])
        self.updateImbalanceParts(part_weight, L_max)
        self.updateCause()


    def updateImbalanceParts(self, part_weight, L_max):
        for p in self.impacted_parts:
            if part_weight[p] > L_max:
                self.imbalance_parts.add(p)
                self.was_imbalance.add(p)
                self.is_imbalance_dst[p] = True # only parts as dst
            else: # if a part becomes balanced, remove
                if self.is_imbalance_dst[p] == True:
                    self.imbalance_parts.discard(p)
                    self.is_imbalance_dst[p] = False

    def updateCause(self):
        for part in self.imbalance_parts:
            if len(self.cause[part]) == 0: # newly generated imbalance part
                if self.has_found_cause[part]:
                    self.exception.add(part) # this part is inherently imbalanced
                else:
                    self.has_found_cause[part] = True
                    m = self.map_dst2move[part]
                    for m_idx in m:
                        self.cause[part].append(m_idx)
                        self.normal_moves.discard(m_idx)

    def solve(self, part_weight, L_max, vertex_weight):
        while len(self.imbalance_parts) > len(self.exception):
            self.impacted_parts.clear()
            for part in self.imbalance_parts:
                if len(self.cause[part]) == 0: # this part is inherently imbalanced
                    continue
                mov_idx = self.cause[part].popleft() # get one possible critical move that may cause imbalance to one part
                self.jettison_moves.append(mov_idx)

                v_weight = vertex_weight[self.moves["v"][mov_idx]]
                src = self.moves["src"][mov_idx]
                dst = self.moves["dst"][mov_idx]
                part_weight[src] += v_weight # roll-back
                part_weight[dst] -= v_weight
                self.impacted_parts.add(src)
                self.impacted_parts.add(dst)

            self.updateImbalanceParts(part_weight, L_max)
            self.updateCause()

        for part in self.was_imbalance:
            for i in self.cause[part]:
                self.vulnerable_moves.append(i)

        max_balance_submove = list()
        for mov_idx in self.normal_moves:
            max_balance_submove.append(mov_idx)
        for mov_idx in self.vulnerable_moves:
            max_balance_submove.append(mov_idx)
        #print("inherently imbalanced parts:", self.exception)
        return max_balance_submove

# test adjusting algorithm
def generate_test_data(num_vertices, num_parts, max_weight, max_L_max):
    """
    random test data
    :param num_vertices: 顶点数量
    :param num_parts: 部分数量
    :param max_weight: 顶点权重的最大值
    :param max_L_max: L_max 的最大值
    :return: vertices, src, dst, part_weight, L_max, vertex_weight
    """
    vertices = list(range(num_vertices))
    src = np.random.randint(0, num_parts, size=num_vertices).tolist()
    dst = np.random.randint(0, num_parts, size=num_vertices).tolist()
    part_weight = np.random.randint(max_L_max - max_weight, max_L_max + max_weight//3, size=num_parts)
    L_max = np.random.randint(max_L_max-5, max_L_max)
    vertex_weight = np.random.randint(1, max_weight, size=num_vertices)
    return vertices, src, dst, part_weight, L_max, vertex_weight

def validate_result(part_weight, L_max):
    """
    check balance
    :param vertices: 顶点列表
    :param src: 源部分列表
    :param dst: 目标部分列表
    :param part_weight: 部分权重
    :param L_max: 最大允许权重
    :param vertex_weight: 顶点权重
    :param result_moves: 算法返回的移动索引
    :return: 是否通过验证
    """
    for weight in part_weight:
        if weight > L_max:
            print("Validation failed: Part weight exceeds L_max.")
            return False

    print("Validation passed: All part weights are within L_max.")
    return True
