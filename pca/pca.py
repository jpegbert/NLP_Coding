import numpy as np


# https://github.com/SeanLee97/nlp_learning/blob/master/pca/pca.py


def PCA(x, n_components=2):
    # 1. 对每个特征（每一行）进行去中心化，即每个数据减去均值
    mean_val = np.mean(x, axis=0)
    mean_x = x - mean_val

    # 2. 求mean_x协方差方阵
    C_x = np.cov(mean_x, rowvar=True)

    # 3. 求C_x特征值和特征向量
    eig_vals, eig_vects = np.linalg.eig(np.mat(C_x))

    # 4. 对特征值从大到小排序
    sorted_idx = np.argsort(-eig_vals) # 返回从大到小排序的下标序列

    # 5. 降维
    topn_index = sorted_idx[:n_components] # 最大的n_components个特征值对应的下标
    topn_vects = eig_vects[topn_index, :] # 得到最大的n_components个特征值对应下标对应的向量

    # 6. 投影到低维空间
    pca_x = topn_vects * x
    return pca_x


if __name__ == '__main__':
    x = np.mat([[-1, -1, 0, 2, 0],
                [-2,  0, 0, 1, 1]])
    x_ = PCA(x, n_components=1)
    print(x_)
