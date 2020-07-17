"""
@leofansq
https://github.com/leofansq
"""
import numpy as np 
import copy

def MSE_multi(wi):
    """
    MSE 多类扩展训练
    Parameters:
        wi : 由多类样本构成的列表[[类1样本],[类2样本],...]
    Return:
        a: 权重矩阵
    """
    # 增广 & 构建label矩阵y
    w_i = copy.deepcopy(wi)
    w = []
    y = np.zeros((len(w_i), len(w_i)*len(w_i[0])))
    for idx, i in enumerate(w_i):
        for j in i: 
            j.append(1)
            w.append(j)
        y[idx, idx*len(w_i[0]):(idx+1)*len(w_i[0])] = 1
    w = np.array(w).T

    # 计算权向量矩阵a
    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(w, w.T)), w), y.T)

    return a

def test(w_test, a):
    """
    测试并计算错误率.
    Parameters:
        w_test: 测试样本集
        a : 权向量矩阵
    Return: 
        f_ratio: 错误率
    """
    w_t = copy.deepcopy(w_test)
    f_cnt = 0
    for idx_i, i in enumerate(w_t):
        for idx_j, j in enumerate(i):
            j.append(1)
            j = np.array(j)
            if np.argmax(np.matmul(a.T, j)) != idx_i: f_cnt += 1

    f_ratio = f_cnt / ((idx_i+1)*(idx_j+1))

    return f_ratio




if __name__ == "__main__":
    w_1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
    w_2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
    w_3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
    w_4 = [[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]]
    # Train
    wi = [w_1[:8], w_2[:8], w_3[:8], w_4[:8]]
    a = MSE_multi(wi)
    print (a)
    # Test
    w_test = [w_1[8:], w_2[8:], w_3[8:], w_4[8:]]
    f_ratio = test(w_test, a)
    print (f_ratio)

    
