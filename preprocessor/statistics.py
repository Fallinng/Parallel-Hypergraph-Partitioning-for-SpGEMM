from numba import cuda
import numpy as np
import math

'''
    概率论分析矩阵pattern函数库
'''

def nzFilter(ptr):
    '''
    non-zero filter
    '''
    num_init = len(ptr) - 1
    nz_elements = [i for i in range(num_init) if ptr[i+1] - ptr[i] > 0]
    nz_num = len(nz_elements)

    return nz_elements, nz_num

@cuda.jit
def computeDiff(elements, N, ptr, avg, diff):
    n = cuda.grid(1)
    if n >= N:
        return
    
    element = elements[n]
    diff[n] = (ptr[element+1] - ptr[element] - avg) **2

@cuda.jit
def reduce_shfl(d_x, d_y, N):
    '''
    x: to be reduced array
    y: reduced array
    N: len(x)
    '''
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n = cuda.grid(1)
    if n >= N:
        return
    # allocate shared memory
    s_y = cuda.shared.array(1024, dtype=np.float32)
    s_y[tid] = d_x[n] if n < N else 0
    cuda.syncthreads()
    # reduce
    i = cuda.blockDim.x >> 1
    while i >= 32:
        if tid < i:
            s_y[tid] += s_y[tid + i]
        cuda.syncthreads()
        i >>= 1
    y = s_y[tid]
    while i > 0:
        if tid < i:
            y += cuda.shfl_down_sync(0xFFFFFFFF, y, i)
        i >>= 1
    if tid == 0:
        d_y[bid] = s_y[0]

def computeSizeVar(elements, N, avg, ptr):
    '''
        effective for N <= 1024*1024
    '''
    block_size = 1024
    grid_size = (N + block_size - 1) // block_size

    d_diff = cuda.device_array(N, dtype=np.float32)
    d_sum_diff = cuda.device_array(grid_size, dtype=np.float32)
    d_elements = cuda.to_device(elements)
    d_ptr = cuda.to_device(ptr)

    computeDiff[grid_size, block_size](d_elements, N, d_ptr, avg, d_diff)
    reduce_shfl[grid_size, block_size](d_diff, d_sum_diff, N)
    reduce_shfl[1, 1024](d_sum_diff, d_sum_diff, grid_size)

    sum_diff = d_sum_diff.copy_to_host()
    return sum_diff[0] / N

# estimate E[connectivity]
def estimateCon(net_num, net_size_bar, w_net_bar, var, k):
    tmp = (var + (net_size_bar**2) - net_size_bar) / k / 2
    return math.ceil(net_num * (net_size_bar - 1 - tmp) * w_net_bar)

def estimateK(cache_size, nnz, 
              net_num, net_size_bar, w_net_bar, var, 
              const_1, const_2):
    a = 2 * cache_size
    b = (-2) * (nnz * (2 + const_1) + const_2 * net_num * w_net_bar * (net_size_bar - 1))
    c = const_2 * net_num * w_net_bar * (var + (net_size_bar**2) - net_size_bar)

    delta = b**2 - 4*a*c
    if delta < 0:
        return -1, -1
    else:
        k1 = (-b - math.sqrt(delta)) / (2*a)
        k2 = (-b + math.sqrt(delta)) / (2*a)
        return math.floor(k1), math.ceil(k2)
    
# estimate hyperedge collision
def estimateNetCollision():
    pass

def testStatistics(name, cache_size, const_1, const_2):
    from fileLoader import loadMatrix
    matrix_path = "matrix_data/" + name + "_CSC.txt"

    _, _, ptr, idx = loadMatrix(matrix_path)
    elements, net_num = nzFilter(ptr)
    net_size_bar = len(idx) / net_num
    var = computeSizeVar(elements, net_num, net_size_bar, ptr)
    _, k2 = estimateK(cache_size, len(idx), net_num, net_size_bar, 1, var, const_1, const_2)
    E_con = -1
    if k2 != -1:
        w_net_bar = len(idx) / (len(ptr) - 1)
        E_con = estimateCon(net_num, net_size_bar, w_net_bar, var, k2)
    
    return k2, E_con, var

if __name__ == "__main__":
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    from contrast import genRandomPartition
    from evaluation import evaluateByPartition
    from fileLoader import loadMatrix
    '''
        test computeSizeVar
    '''
    names = ["wiki-Vote", "m133-b3", "mario002", "scircuit", "2cubes_sphere", "offshore", 
             "filter3D", "mac_econ_fwd500", "email-Enron", "t2dah_a", "af23560"]
    '''names = ["Ge87H76", "gupta2", "webbase-1M", "cop20k_A"]'''
    name = "2cubes_sphere"
    

    # estimate k
    cache_size = 64000 / 8
    const_1 = 2
    const_2 = 0.5

    length = len(names)
    result = {"var": [], "k": [], "E_con": [], "con": [], "accuracy": []}
    for name in names:
        k, E_con, var = testStatistics(name, cache_size, const_1, const_2)
        actual_con = -1
        if k != -1:
            matrix_path = "matrix_data/" + name + "_CSR.txt"
            _, _, ptr, idx = loadMatrix(matrix_path)
            random_parts = genRandomPartition(ptr, k)
            _, actual_con = evaluateByPartition(random_parts, ptr, idx)
        result["var"].append(var)
        result["k"].append(k)
        result["E_con"].append(E_con)
        result["con"].append(actual_con)
        accuracy = E_con / actual_con * 100 if actual_con != -1 else -1
        result["accuracy"].append(accuracy)

    for cata, value in result.items():
        if cata == "var" or cata == "accuracy":
            value = [f"{v:.4f}" for v in value]  # 保留 4 位小数
        print(f"{cata}: {value}")