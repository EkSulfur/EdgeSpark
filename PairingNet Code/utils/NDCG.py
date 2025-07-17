import numpy as np

def dcg_at_k(scores, k):
    """计算 DCG@k 的值"""
    return np.sum(scores[:k] / np.log2(np.arange(2, k + 2)))

def ndcg_at_k(scores, k):
    """计算 NDCG@k 的值"""
    best = dcg_at_k(sorted(scores, reverse=True), k)
    if best == 0:
        return 0
    return dcg_at_k(scores, k) / best

def ndcg(golden, current, n=-1):
    '''
    golden : 理想排序下的1-0矩阵
    current : 真实排序下的1-0矩阵
    n : top n
    '''
    log2_table = np.log2(np.arange(2, 2000))

    def dcg_at_n(rel, n):
        rel = np.asfarray(rel)[:n]
        dcg = np.sum(np.divide(np.power(2, rel) - 1, log2_table[:rel.shape[0]]))
        return dcg

    ndcgs = []
    for i in range(len(current)):
        k = len(current[i]) if n == -1 else n
        idcg = dcg_at_n(sorted(golden[i], reverse=True), n=k)
        dcg = dcg_at_n(current[i], n=k)
        tmp_ndcg = 0 if idcg == 0 else dcg / idcg
        ndcgs.append(tmp_ndcg)
    SUM_ndcgs = sum(ndcgs)
    return 0. if len(ndcgs) == 0 else sum(ndcgs) / (len(ndcgs))


def remove_zero_rows(array):
    # 使用numpy的all和axis参数来检查每一行是否全为0
    mask = np.all(array == 0, axis=1)
    # 使用mask来从array中删除全为0的行
    new_array = array[~mask]
    return new_array


def calculate_NDCG(sim_matrix, GT_pairs):
    ##################
    # NDCG 指标
    ##################
    idx = np.argsort(-sim_matrix, axis=1)
    length = sim_matrix.shape[0]
    gt = np.zeros((length, length), dtype=np.uint8)
    pred = np.zeros((length, length), dtype=np.uint8)
    l = []
    for i in range(len(GT_pairs)):
        location = np.argwhere(idx[GT_pairs[i][0]] == GT_pairs[i][1])[0][0]
        gt[GT_pairs[i][0], GT_pairs[i][1]] = 1
        pred[GT_pairs[i][0], location] = 1

    new_gt = remove_zero_rows(gt)
    new_pred = remove_zero_rows(pred)
    NDCG_result = (ndcg(new_gt, new_pred, 5), ndcg(new_gt, new_pred, 10), ndcg(new_gt, new_pred, 20))

    print("NDCG@5, 10, 20")
    print(NDCG_result)

if __name__ == "__main__":
    # 假设我们有一个形状为 (20, 10) 的搜索结果矩阵
    results = np.random.randint(0, 5, size=(20, 10))

    # 计算每个检索项的 NDCG@10 值
    ndcg_scores = [ndcg_at_k(row, 10) for row in results]

    # 计算所有检索项的平均 NDCG@10 值
    mean_ndcg = np.mean(ndcg_scores)

    print(f"Mean NDCG@10: {mean_ndcg:.3f}")