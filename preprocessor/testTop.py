from partitioner import Partitioner
from evaluation import evaluateByArr
from fileLoader import loadMatrix

def testTop(name, k, epsilon_init, epsilon, max_iter, batch_size, mute=False):
    matrix_path = [None, None]
    matrix_path[0] = "matrix_data/" + name + "_CSR.txt"
    matrix_path[1] = "matrix_data/" + name + "_CSC.txt"
    
    _, _, ptr_v2n, idx_v2n = loadMatrix(matrix_path[0])
    _, _, ptr_n2v, _ = loadMatrix(matrix_path[1])
    ptr_B = ptr_v2n
    
    print(name)
    # initialize partitioner
    t_start_init = time.perf_counter()
    u_part = Partitioner(ptr_B, 
                         ptr_v2n, idx_v2n, ptr_n2v, 
                         k, epsilon_init, 
                         epsilon, max_iter)
    t_end_init = time.perf_counter()
    arr_end = u_part.getArrEnd()
    pingpong = u_part.getPingpong()
    cut_net0, con0 = evaluateByArr(arr_end, ptr_n2v, ptr_v2n, pingpong)
    if not mute:
        print(f"Initial: ({cut_net0}, {con0})")

    # solve test
    t_solve_start = time.perf_counter()
    u_part.solve(batch_size)
    t_solve_end = time.perf_counter()

    print(f"(initialization time, solve time) = ({t_end_init - t_start_init:.4f}, {t_solve_end - t_solve_start:.4f})")

    arr_end = u_part.getArrEnd()
    pingpong = u_part.getPingpong()
    cut_net, con = evaluateByArr(arr_end, ptr_n2v, ptr_v2n, pingpong)
    if not mute:
        print(f"Refined: ({cut_net}, {con})")

    con_arr = [con0, con]
    return con_arr

def testBatchSizeAndTimeCost(name, k, epsilon_init, epsilon, max_iter, batch_size):
    matrix_path = [None, None]
    matrix_path[0] = "matrix_data/" + name + "_CSR.txt"
    matrix_path[1] = "matrix_data/" + name + "_CSC.txt"
    
    _, _, ptr_v2n, idx_v2n = loadMatrix(matrix_path[0])
    _, _, ptr_n2v, _ = loadMatrix(matrix_path[1])
    ptr_B = ptr_v2n

    u_part = Partitioner(ptr_B, 
                         ptr_v2n, idx_v2n, ptr_n2v, 
                         k, epsilon_init, 
                         epsilon, max_iter)
    
    t_solve_start = time.perf_counter()
    u_part.solve(batch_size)
    t_solve_end = time.perf_counter()

    return t_solve_end - t_solve_start

if __name__ == "__main__":

    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    import time
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    '''
        test Partitioner
    '''

    '''names = ["wiki-Vote", "m133-b3", "mario002", "scircuit", "2cubes_sphere", "offshore", 
         "filter3D", "mac_econ_fwd500", "p2p-Gnutella31", "email-Enron"]'''
    '''names = ["wiki-Vote", "m133-b3", "mario002", "scircuit", "2cubes_sphere", "offshore", 
             "filter3D", "mac_econ_fwd500", "email-Enron", "t2dah_a", "af23560"]'''
    names = ["mario002"]
    name = "m133-b3"
    k = 1024
    epsilon_init = 0.02
    epsilon = 0.02
    max_iter = 1
    batch_size = 4000

    mute = True

    '''con_list = {"Initial": [], "Refined": []}
    for name in names:
        con_arr = testTop(name, k, epsilon_init, epsilon, max_iter, batch_size, mute)
        con_list["Initial"].append(con_arr[0])
        con_list["Refined"].append(con_arr[1])

    for src, con in con_list.items():
        print(f"{src}: {con}")'''

    # test batch size and time cost
    '''size = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]'''
    size = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    time_cost = []
    for batch_size in size:
        time_cost.append(testBatchSizeAndTimeCost(name, k, epsilon_init, epsilon, max_iter, batch_size))
    print(f"batch_size: {size}")
    print(f"time_cost: {[round(t, 4) for t in time_cost]}")
