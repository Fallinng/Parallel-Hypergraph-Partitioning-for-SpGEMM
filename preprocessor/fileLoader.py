from collections import defaultdict

def loadMatrix(matrix_path):
    """加载CSR/CSC格式文件，适配mtx_processor的输出"""
    format, shape, indptr, indices = None, None, [], []
    with open(matrix_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[Format]'):
                format = line.split()[1]
            elif line.startswith('[Shape]'):
                shape = tuple(map(int, line.split()[1:3]))
            elif line.startswith('[Indptr]'):
                indptr = list(map(int, line.split()[1:]))
            elif line.startswith('[Indices]'):
                indices = list(map(int, line.split()[1:]))
    return format, shape, indptr, indices

def loadPartition(partition_path):
    '''加载分区'''
    parts = defaultdict(list)
    with open(partition_path, 'r') as f:
        i = 0
        for line in f:
            if ':' in line:
                _, rows = line.strip().split(':')
                if rows:
                    parts[i] = list(map(int, rows.split(',')))
                else:
                    parts[i] = []
                i += 1
    return parts

# 保存 seq 到文件
def saveSeqence(seq, path):
    with open(path, "w") as f:
        f.write(",".join(map(str, seq)))  # 用逗号拼接元素并写入文件

# 从文件加载 seq
def loadSequence(path):
    with open(path, "r") as f:
        content = f.read().strip()  # 读取文件内容并去除首尾空白
        seq = list(map(int, content.split(",")))  # 按逗号分割并转换为整数列表
    return seq
