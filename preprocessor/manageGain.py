import numpy as np

def sortByGain(indices, vertices, src_arr, dst_arr, gains):
    # 获取 gains 数组的降序排序索引
    sorted_indices = np.argsort(gains)[::-1]  # [::-1] 用于将排序结果反转为降序

    # 根据排序索引重新排列所有数组
    indices = indices[sorted_indices]
    vertices = vertices[sorted_indices]
    src_arr = src_arr[sorted_indices]
    dst_arr = dst_arr[sorted_indices]

    return indices, vertices, src_arr, dst_arr

def getAppliedIndices(sorted_idx, submove_idx):
    applied_idx = []
    for i in range(len(submove_idx)):
        applied_idx.append(sorted_idx[submove_idx[i]])
    return applied_idx

