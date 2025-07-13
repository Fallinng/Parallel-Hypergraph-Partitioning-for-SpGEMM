from numba import cuda
import numpy as np

from hpgUpdate import heapifyMin

@cuda.jit
def findDstParts(vertices, src_arr, dst_arr, gains, batch_size, 
                 w_net, 
                 ptr_v2n, idx_v2n, 
                 ptr_n2p, arr, arr_end, pingpong, 
                 ptr_comp, merge_heap):
    '''
        find destination part of highest gain
    '''
    n = cuda.grid(1)
    if n >= batch_size:
        return
    v = vertices[n]
    src = src_arr[n]
    start, end = ptr_v2n[v], ptr_v2n[v + 1]
    offset = start # offset of the merge_heap
    # initialize ptr_comp that points to k sorted arrays
    for i in range(start, end):
        ptr_comp[i] = ptr_n2p[idx_v2n[i]]
    # initialize merge_heap for k-way merging
    size = 0
    for i in range(start, end):
        net = idx_v2n[i]
        sel = pingpong[net]
        if ptr_comp[i] >= arr_end[sel, net]: # invalid
            continue
        merge_heap[offset + size, 0] = arr[sel, ptr_comp[i], 0] # part_id
        merge_heap[offset + size, 1] = i # ptr_comp[i]'s position
        heapifyMin(merge_heap, offset, size + 1, size, 1) # element added to the bottom, sift-up
        size += 1
        ptr_comp[i] += 1

    # output elements in k-arrays in order to compute gain
    pid, pos_ptr = -1, -1
    tmp = cuda.local.array(2, dtype=np.int32)
    tmp[0], tmp[1] = -1, -1 # represents pid, gain
    max_gain_pid, max_gain = -1, -1
    while size > 0:
        pid = merge_heap[offset, 0]
        pos_ptr = merge_heap[offset, 1]

        # update merge_heap
        net = idx_v2n[pos_ptr]
        sel = pingpong[net]
        tmp_ptr_comp = ptr_comp[pos_ptr]
        val = arr[sel, tmp_ptr_comp - 1, 1]
        if tmp_ptr_comp >= arr_end[sel, net]: # no more valid elements in this array
            merge_heap[offset, 0] = merge_heap[offset + size - 1, 0]
            merge_heap[offset, 1] = merge_heap[offset + size - 1, 1]
            heapifyMin(merge_heap, offset, size - 1, 0, -1) # sift-down from relative position 0
            size -= 1
        else:
            merge_heap[offset, 0] = arr[sel, tmp_ptr_comp, 0]
            ptr_comp[pos_ptr] += 1 # move to next element in the array
            heapifyMin(merge_heap, offset, size, 0, 1)

        # compute gain for each pid
        if tmp[0] != pid: # when heap streams out a new pid, conclude the gain of the previous pid
            if tmp[0] != -1 and tmp[1] > max_gain:
                max_gain_pid = tmp[0]
                max_gain = tmp[1]
            tmp[0] = pid
            tmp[1] = 0 if (pid == src and val == 1) else w_net[net]
        else:
            tmp[1] += 0 if (pid == src and val == 1) else w_net[net]
    if tmp[0] != -1 and tmp[1] > max_gain:
        max_gain_pid = tmp[0]
        max_gain = tmp[1]

    dst_arr[n] = max_gain_pid
    gains[n] = max_gain
