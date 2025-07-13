from numba import cuda
import numpy as np

@cuda.jit(device=True)
def heapifyMin(heap, offset, size, pos, direction):
    '''
        heapify operation for Min-Heap
        offset - when heaps share an array, offset is the starting position of the heap
        size   - current size of the heap
        pos    - position (relative to offset) of the element needed to be adjusted
        direction - 1: sift up, -1: sift down
    '''
    if direction == 1:
        while pos > 0:
            parent = (pos - 1) // 2
            if heap[offset + pos, 0] >= heap[offset + parent, 0]:
                break
            # 交换元素
            for col in range(heap.shape[1]):  # 遍历列，交换所有字段
                heap[offset + pos, col], heap[offset + parent, col] = (
                    heap[offset + parent, col], heap[offset + pos, col]
                )
            pos = parent
    else:
        while True:
            min_pos = pos
            left = pos*2 + 1
            right = pos*2 + 2
            if left < size and heap[offset + left, 0] < heap[offset + min_pos, 0]:
                min_pos = left
            if right < size and heap[offset + right, 0] < heap[offset + min_pos, 0]:
                min_pos = right
            if min_pos == pos:
                break
            # 交换元素
            for col in range(heap.shape[1]):  # 遍历列，交换所有字段
                heap[offset + pos, col], heap[offset + min_pos, col] = (
                    heap[offset + min_pos, col], heap[offset + pos, col]
                )
            pos = min_pos

'''
    basic device functions
'''
@cuda.jit(device=True)
def atomicApplyChange2Buffer(e, src, dst, 
                             buf, ptr_buf):
    '''
    apply vertex move in the perspective of hyperedges' pin number change in parts
    e - hyperedge id
    src - source part id
    dst - destination part id
    '''
    if src == None: # when initializing partition, src is None
        pos0 = cuda.atomic.add(ptr_buf, e, 1) # allocate 1 position for dst change
        buf[pos0, 0], buf[pos0, 1] = dst, 1
    else:
        pos0 = cuda.atomic.add(ptr_buf, e, 2) # allocate 2 positions for src and dst change
        pos1 = pos0 + 1
        buf[pos0, 0], buf[pos0, 1] = src, -1
        buf[pos1, 0], buf[pos1, 1] = dst, 1

@cuda.jit(device=True)
def heapifyBuffer2Heap(e, heap, offset, heap_size,
                       buf, ptr_buf, buf_offset):
    '''
    transfer data from buffer to heap, every function process one hyperedge
    e - hyperedge id
    heap - heap array
    offset - heap start position in heap array
    '''
    ptr_buf_cur = ptr_buf[e] # from backward to the start offset of the buffer
    size = 0
    while ptr_buf_cur > buf_offset:
        # 1. get the first element in buffer
        ptr_buf_cur -= 1
        abs_pos = size + offset
        heap[abs_pos, 0], heap[abs_pos, 1] = buf[ptr_buf_cur, 0], buf[ptr_buf_cur, 1]
        heapifyMin(heap, offset, size + 1, size, 1)
        size += 1

    heap_size[e] = size
    ptr_buf[e] = ptr_buf_cur

@cuda.jit(device=True)
def mergeHeapAndArray(e, heap, offset, heap_size,
                      arr, start, arr_end, pingpong):
    '''
    arr - 3D array, 2 * nnz * 2
    '''
    sel = pingpong[e] # 0 or 1, indicating the valid first dimension of arr at present, then merge result to arr[1-sel]

    ptr_arr = cuda.local.array(2, dtype=np.int32)
    ptr_arr[0], ptr_arr[1] = start, start

    ptr_arr_end = cuda.local.array(2, dtype=np.int32)
    ptr_arr_end[0], ptr_arr_end[1] = arr_end[0, e], arr_end[1, e]

    size = heap_size[e]
    idx, val = -1, -1
    tmp = cuda.local.array(2, dtype=np.int32)
    tmp[0], tmp[1] = -1, -1

    while size > 0 or ptr_arr[sel] < ptr_arr_end[sel]:
        arr_key_is_smaller = (arr[sel, ptr_arr[sel], 0] < heap[offset, 0] and ptr_arr[sel] < ptr_arr_end[sel]) or (size == 0)
        if arr_key_is_smaller:
            idx = arr[sel, ptr_arr[sel], 0]
            val = arr[sel, ptr_arr[sel], 1]
            ptr_arr[sel] += 1
        else:
            idx = heap[offset, 0]
            val = heap[offset, 1]
            heap[offset, 0], heap[offset, 1] = heap[offset + size - 1, 0], heap[offset + size - 1, 1]
            heapifyMin(heap, offset, size - 1, 0, -1)
            size -= 1
        
        if tmp[0] != idx:
            if tmp[0] != -1 and tmp[1] != 0:
                arr[1 - sel, ptr_arr[1 - sel], 0], arr[1 - sel, ptr_arr[1 - sel], 1] = tmp[0], tmp[1]
                ptr_arr[1 - sel] += 1
            tmp[0], tmp[1] = idx, val
        else:
            tmp[1] += val
    if tmp[0] != -1 and tmp[1] != 0:
        arr[1 - sel, ptr_arr[1 - sel], 0], arr[1 - sel, ptr_arr[1 - sel], 1] = tmp[0], tmp[1]
        ptr_arr[1 - sel] += 1
    
    arr_end[1 - sel, e] = ptr_arr[1 - sel] # set actual valid end position
    arr_end[sel, e] = ptr_arr_end[1 - sel] # set to the end of the valid array
    heap_size[e] = size
    pingpong[e] = 1 - sel

'''
    initialization functions
'''
@cuda.jit
def initHpgBuf(elements, N, dst_parts,
               ptr_v2n, idx_v2n,
               buf, ptr_buf, 
               bitmap):
    n = cuda.grid(1)
    if n >= N:
        return
    v = elements[n]
    dst = dst_parts[n]

    start_net = ptr_v2n[v]
    end_net = ptr_v2n[v + 1]
    for net in idx_v2n[start_net:end_net]:
        atomicApplyChange2Buffer(net, None, dst, buf, ptr_buf)
        # mark the net in bitmap by atomic operation
        cuda.atomic.or_(bitmap, net // 32, 1 << (31 - (net % 32)))

@cuda.jit
def initHpgArr(N, 
               buf, ptr_buf, ptr_buf_offset,
               heap, heap_offset, heap_size, 
               ptr_n2p, arr, arr_end, pingpong, 
               bitmap):
    '''
    initialize HyperGraph net-to-part structure step2
    '''
    net = cuda.grid(1)
    if net >= N:
        return
    bits = bitmap[net // 32]
    # clear bitmap
    if net % 32 == 0:
        bitmap[net // 32] = 0
    if (bits & (1 << (31 - (net % 32)))) == 0:
        return
    # heapify buffer to heap
    heapifyBuffer2Heap(net, heap, heap_offset[net], heap_size,
                       buf, ptr_buf, ptr_buf_offset[net])
    # merge heap and array
    mergeHeapAndArray(net, heap, heap_offset[net], heap_size,
                       arr, ptr_n2p[net], arr_end, pingpong)

def initHpg(parts, v_num, n_num,
            ptr_v2n, idx_v2n, 
            buf, ptr_buf, ptr_buf_offset, 
            heap, heap_offset, heap_size, 
            ptr_n2p, arr, arr_end, pingpong, 
            bitmap):
    '''
    initialize HyperGraph net-to-part structure step1
    '''
    host_nz_rows = np.empty(v_num, dtype=np.int32)
    host_dst_parts = np.empty(v_num, dtype=np.int32)
    i = 0
    for pid, vertices in parts.items():
        for v in vertices:
            host_nz_rows[i] = v
            host_dst_parts[i] = pid
            i += 1
    
    nz_rows = cuda.to_device(host_nz_rows)
    num_nz_rows = i
    dst_parts = cuda.to_device(host_dst_parts)

    block_size = 128
    grid_size = (num_nz_rows + block_size - 1) // block_size

    # initialize buffer
    initHpgBuf[grid_size, block_size](nz_rows, num_nz_rows, dst_parts,
                                      ptr_v2n, idx_v2n,
                                      buf, ptr_buf, 
                                      bitmap)

    N = n_num
    grid_size = (N + block_size - 1) // block_size
    # initialize array
    initHpgArr[grid_size, block_size](N, 
                                      buf, ptr_buf, ptr_buf_offset,
                                      heap, heap_offset, heap_size, 
                                      ptr_n2p, arr, arr_end, pingpong,
                                      bitmap)
    
'''
    updating functions
'''
@cuda.jit
def updateBuf(applied_idx, num_applied, 
              vertices, src_arr, dst_arr, 
              ptr_v2n, idx_v2n, 
              buf, ptr_buf, 
              bitmap, 
              warp_size):
    '''
        use a warp of threads to update a vertex's hyperedge status to buffer
    '''
    n = cuda.grid(1)
    lane_id = n % warp_size
    n = n // warp_size
    if n >= num_applied:
        return
    mov_idx = applied_idx[n]
    v, src, dst = vertices[mov_idx], src_arr[mov_idx], dst_arr[mov_idx]
    start_net = ptr_v2n[v]
    end_net = ptr_v2n[v + 1]
    for i in range(start_net + lane_id, end_net, warp_size):
        net = idx_v2n[i]
        atomicApplyChange2Buffer(net, src, dst, buf, ptr_buf)
        # mark the net in bitmap by atomic operation
        cuda.atomic.or_(bitmap, net // 32, 1 << (31 - (net % 32)))
    pass

@cuda.jit
def updateArr(buf, ptr_buf, ptr_buf_offset, 
              heap, heap_offset, heap_size, 
              ptr_n2p, arr, arr_end, pingpong, 
              bitmap, len_bitmap):
    '''
        a single thread check 32-bit of the bitmap and update corresponding marked hyperedge 
    '''
    n = cuda.grid(1)
    if n >= len_bitmap:
        return
    bits = bitmap[n]
    end = (n + 1) * 32
    while bits > 0:
        first_bit = cuda.ffs(bits)
        net = end - first_bit
        bits &= ~(1 << (first_bit - 1)) # clear the first set bit
        # update heap and array
        heapifyBuffer2Heap(net, heap, heap_offset[net], heap_size,
                           buf, ptr_buf, ptr_buf_offset[net])
        mergeHeapAndArray(net, heap, heap_offset[net], heap_size,
                           arr, ptr_n2p[net], arr_end, pingpong)
    bitmap[n] = 0 # clear the bitmap

def applyMoves(h_applied_idx, num_applied, 
               vertices, src_arr, dst_arr,
               ptr_v2n, idx_v2n, 
               buf, ptr_buf, ptr_buf_offset, 
               heap, heap_offset, heap_size, 
               ptr_n2p, arr, arr_end, pingpong, 
               bitmap, len_bitmap):
    if num_applied == 0:
        return
    
    applied_idx = cuda.to_device(h_applied_idx)
    block_size = 128
    warp_size = 32
    if num_applied > 2048:
        warp_size = 16
    grid_size = (num_applied * warp_size + block_size - 1) // block_size
    # update buffer
    updateBuf[grid_size, block_size](applied_idx, num_applied,
                                     vertices, src_arr, dst_arr,
                                     ptr_v2n, idx_v2n,
                                     buf, ptr_buf, 
                                     bitmap, 
                                     warp_size)
    # update 3D-array
    grid_size = (len_bitmap + block_size - 1) // block_size
    updateArr[grid_size, block_size](buf, ptr_buf, ptr_buf_offset,
                                     heap, heap_offset, heap_size,
                                     ptr_n2p, arr, arr_end, pingpong,
                                     bitmap, len_bitmap)

    