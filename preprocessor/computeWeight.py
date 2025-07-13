import numpy as np

'''
    transfer compressed format matrix to HyperGraph expression
'''
def getNetWeight(weight, N, ptr_B):
    for n in range(N):
        weight[n] = ptr_B[n+1] - ptr_B[n]

def getVertexWeight(vertex_weight, N, ptr, idx, net_weight):
    for n in range(N):
        start = ptr[n]
        end = ptr[n+1]
        for i in range(start, end):
            vertex_weight[n] += net_weight[idx[i]]

def getSumWeight(vertex_weight):
    return sum(vertex_weight)

def getMaxWeight(sum_weight, epsilon, k):
    '''
        compute balance constraint
        L_max <= (1 + epsilon) * sumweight / k
    '''
    return (1 + epsilon) * sum_weight / k