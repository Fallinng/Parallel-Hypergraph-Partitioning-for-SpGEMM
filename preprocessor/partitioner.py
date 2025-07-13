from initialPartition import InitialPartition
from hpgUpdate import initHpg, applyMoves
from scheduler import Scheduler
from computeWeight import getNetWeight, getVertexWeight
from statistics import nzFilter
from balancing import maxBalanceSubmoveFinder
from parallelRefinement import findDstParts
from manageGain import sortByGain, getAppliedIndices

import numpy as np
from numba import cuda
import math
from collections import defaultdict

class Partitioner:
    def __init__(self, B_ptr, 
                 ptr_v2n, idx_v2n, ptr_n2v, 
                 k, epsilon_init, epsilon,
                 max_iter):
        '''
        Compressed Format
            ptr, idx

        Partition Constraints:
            k - number of partitions
            epsilon_init - imbalance ratio for initial partitioning
                epsilon - imbalance ratio for parallel refinement
        '''
        N = len(ptr_v2n) - 1 # number of vertices
        M = len(ptr_n2v) - 1 # number of nets
        nnz = len(idx_v2n) # number of non-zero elements
        self.max_iter = max_iter
        self.k = k

        # initialize vertex weights & weight constraint
        w_net = np.zeros(M, dtype=np.int32)
        getNetWeight(w_net, M, B_ptr)
        self.d_w_net = cuda.to_device(w_net)
        self.w_vertices = np.zeros(N, dtype=np.int32)
        getVertexWeight(self.w_vertices, N, ptr_v2n, idx_v2n, w_net)
        self.L_max = (1 + epsilon) * sum(self.w_vertices) / k

        # initialize part to vertex mapping - p2v
        nz_vertices, _ = nzFilter(ptr_v2n)
        L_max_init = (1 + epsilon_init) * sum(self.w_vertices) / k
        ip = InitialPartition(nz_vertices, self.w_vertices, L_max_init, k)
        self.parts = ip.getInitialPartition() # transfer to sets later
        self.w_part = ip.getInitialPartWeight()
        self.unassigned = ip.getIgnoredVertices()
        #print(f"Unassigned vertices: {len(self.unassigned)}")
        
        save_path = "initial_seq_parts/init_part.txt"
        ip.saveToFile(save_path)
        #print("Initial partitioning completed")

        # assign device arrays for GPU
        '''use bitmap to mark changed hyperedges'''
        bitmap = np.zeros(math.ceil(M / 32), dtype=np.uint32)
        self.len_bitmap = len(bitmap)
        self.d_bitmap = cuda.to_device(bitmap)

        '''original ptr and idx'''
        self.d_ptr_v2n = cuda.to_device(ptr_v2n)
        self.d_idx_v2n = cuda.to_device(idx_v2n)

        ptr_buf = np.zeros(M, dtype=np.int32)
        for i in range(M):
            ptr_buf[i] = 2 * ptr_n2v[i]

        '''buffer & buffer ptrs'''
        self.d_buf = cuda.device_array((nnz*2, 2), dtype=np.int32)
        self.d_ptr_buf = cuda.to_device(ptr_buf)
        self.d_ptr_buf_offset = cuda.to_device(ptr_buf)

        '''heap & heap ptrs'''
        self.d_heap = cuda.device_array((nnz*2, 2), dtype=np.int32)
        self.d_heap_offset = cuda.to_device(ptr_buf)
        self.d_heap_size = cuda.device_array(M, dtype=np.int32)

        '''3D array & ptrs'''
        pingpong = np.ones(M, dtype=np.int32)
        arr_end = np.zeros((2, M), dtype=np.int32)
        for i in range(M):
            arr_end[0, i] = ptr_n2v[i + 1]
            arr_end[1, i] = ptr_n2v[i]
        self.d_ptr_n2p = cuda.to_device(ptr_n2v)
        self.d_arr = cuda.device_array((2, nnz, 2), dtype=np.int32)
        self.d_arr_end = cuda.to_device(arr_end)
        self.d_pingpong = cuda.to_device(pingpong)

        '''gain calculation'''
        self.d_ptr_comp = cuda.device_array(nnz, dtype=np.int32)
        self.d_merge_heap = cuda.device_array((nnz, 2), dtype=np.int32)

        # initialize net to part mapping - n2p
        initHpg(self.parts, N, M, 
                self.d_ptr_v2n, self.d_idx_v2n, 
                self.d_buf, self.d_ptr_buf, self.d_ptr_buf_offset, 
                self.d_heap, self.d_heap_offset, self.d_heap_size,
                self.d_ptr_n2p, self.d_arr, self.d_arr_end, self.d_pingpong, 
                self.d_bitmap)
        print("n2p mapping completed")
         
        print("Partitioner successfully initialized")
        pass

    def getArr(self):
        return self.d_arr.copy_to_host()
    def getArrEnd(self):
        return self.d_arr_end.copy_to_host()
    def getPingpong(self):
        return self.d_pingpong.copy_to_host()
    def getBitmap(self):
        return self.d_bitmap.copy_to_host()
    def getPartitions(self):
        return self.parts.copy()
    
    def solve(self, required_batch_size):
        iter = 0
        # turn parts to sets for quick move application
        local_parts = defaultdict(set)
        for pid, vertices in self.parts.items():
            local_parts[pid] = set(vertices)
        while(iter < self.max_iter):
            # initialize scheduler
            scheduler = Scheduler(self.parts)
            batch, _ = scheduler.getVertexBatch(required_batch_size)

            # assign device arrays for each iteration
            d_dst = cuda.device_array(batch["size"], dtype=np.int32)
            d_gains = cuda.device_array(batch["size"], dtype=np.int32)

            # count applied moves
            applied_move_cnt = 0

            if batch["size"] == 0:
                print("No Valid Vertices in HyperGraph")
                return
            
            while batch["size"] > 0:
                d_vertices = cuda.to_device(batch["v"])
                d_src = cuda.to_device(batch["src"])

                # compute destinations and store to d_dst, to be completed
                block_size = 32
                grid_size = math.ceil(batch["size"] / block_size)
                findDstParts[block_size, grid_size](d_vertices, d_src, d_dst, d_gains, batch["size"], 
                                                    self.d_w_net, 
                                                    self.d_ptr_v2n, self.d_idx_v2n,
                                                    self.d_ptr_n2p, self.d_arr, self.d_arr_end, self.d_pingpong, 
                                                    self.d_ptr_comp, self.d_merge_heap)

                h_dst = d_dst.copy_to_host()
                h_gains = d_gains.copy_to_host()

                # sort gains in descending order
                length = batch["size"]
                indices = np.array(range(length))
                sorted_idx, sorted_vertices, sorted_src, sorted_dst = sortByGain(indices, 
                                                                                 np.array(batch["v"]), np.array(batch["src"]), 
                                                                                 h_dst[0: length], h_gains[0: length])

                # balance control
                for i in range(batch["size"]): # update w_part by assuming all moves are applied
                    w_v = self.w_vertices[batch["v"][i]]
                    src, dst = batch["src"][i], h_dst[i]
                    self.w_part[src] -= w_v
                    self.w_part[dst] += w_v
                submove_finder = maxBalanceSubmoveFinder(sorted_vertices, sorted_src, sorted_dst, 
                                                         self.w_part, self.L_max, self.k)
                max_submove = submove_finder.solve(self.w_part, self.L_max, self.w_vertices)

                # gain management
                applied_idx = getAppliedIndices(sorted_idx, max_submove)
                num_applied = len(applied_idx)
                applied_move_cnt += num_applied

                # apply moves
                applyMoves(applied_idx, num_applied, 
                           d_vertices, d_src, d_dst, 
                           self.d_ptr_v2n, self.d_idx_v2n, 
                           self.d_buf, self.d_ptr_buf, self.d_ptr_buf_offset, 
                           self.d_heap, self.d_heap_offset, self.d_heap_size, 
                           self.d_ptr_n2p, self.d_arr, self.d_arr_end, self.d_pingpong, 
                           self.d_bitmap, self.len_bitmap)
                for i in range(num_applied):
                    src = batch["src"][applied_idx[i]]
                    dst = h_dst[applied_idx[i]]
                    v = batch["v"][applied_idx[i]]
                    local_parts[src].remove(v)
                    local_parts[dst].add(v)
                
                # generate next batch
                batch, _ = scheduler.getVertexBatch(required_batch_size)
            # update parts at list format
            for pid, vertices in local_parts.items():
                self.parts[pid] = list(vertices)
            iter += 1
            #print(f"Iteration {iter} completed, applied moves: {applied_move_cnt}")
    
