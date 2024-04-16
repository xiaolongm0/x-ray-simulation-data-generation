import numpy as np

def find_continuous_regions(ring_img):
    n_cols = ring_img.shape[1]  # 矩阵的列数
    continuous_regions = []
    start_idx = None

    for col_idx in range(n_cols):
        # 检查当前列中非零元素的个数
        nonzero_count = np.count_nonzero(ring_img[:, col_idx])

        if nonzero_count > 0:
            if start_idx is None:
                # 标记连续区域的开始
                start_idx = col_idx
        else:
            if start_idx is not None:
                # 结束连续区域并检查
                if validate_region(ring_img[:, start_idx:col_idx]):
                    continuous_regions.append((start_idx, col_idx - 1))
                # 重置开始索引
                start_idx = None
    
    # 检查结束时是否还在连续区域中
    if start_idx is not None and validate_region(ring_img[:, start_idx:n_cols]):
        continuous_regions.append((start_idx, n_cols - 1))
    
    return continuous_regions

def validate_region(region):
    # 遍历区域的每一列
    min_nonzero_count = np.inf  # 初始化最小非零计数为无穷大
    for col in region.T:  # 转置以遍历列
        nonzero_count = np.count_nonzero(col)
        if nonzero_count < min_nonzero_count:
            min_nonzero_count = nonzero_count

    # 如果任何列的非零元素少于5，返回False
    return min_nonzero_count >= 5

# 示例使用
ring_img = np.random.randint(0, 2, size=(10, 20))  # 创建一个示例矩阵
regions = find_continuous_regions(ring_img)
print("连续区域的起始和结束列索引：", regions)





def find_continuous_regions(ring_img):
    n_cols = ring_img.shape[1]
    continuous_regions = []
    in_region = False
    start_idx = None

    for col_idx in range(n_cols):
        count = np.sum(ring_img[:, col_idx] > 0)
        
        if count >= 5:
            if not in_region:
                in_region = True
                start_idx = col_idx
        else:
            if in_region:
                continuous_regions.append((start_idx, col_idx - 1))
                in_region = False
    
    if in_region:
        continuous_regions.append((start_idx, n_cols - 1))
    
    return continuous_regions