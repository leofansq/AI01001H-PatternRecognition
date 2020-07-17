"""
@leofansq
https://github.com/leofansq
"""
import numpy as np
import copy

def samples_trans(w1, w2):
    """
    规范化增广样本
    Parameters:
        w1: 类1样本
        w2: 类2样本
    Return:
        w: 规范化增广样本
    """
    # 复制样本,防止后续操作改变原始样本
    w_1 = copy.deepcopy(w1)
    w_2 = copy.deepcopy(w2)

    # 增广
    for i in w_1: i.append(1)
    for i in w_2: i.append(1)

    # 规范化
    w_1 = np.array(w_1)
    w_2 = np.array(w_2)
    w_2 = -w_2
    w = np.concatenate([w_1, w_2])

    return w

def HK_algorithm(w1, w2):
    """
    Ho-Kashyap Algorithm
    Parameters:
        w1: 类1样本
        w2: 类2样本
    Return:
        a, b, k
    """
    # 规范化增广样本
    w = samples_trans(w1, w2)

    # 初始化
    a = np.zeros_like(w[1])
    b = np.zeros(w.shape[0]) + 0.5
    yita = 0.5
    th_b = np.zeros(w.shape[0]) + 1e-3
    th_k = 10000
    k = 1

    # 迭代
    while k <= th_k:
        e = np.matmul(w, a) - b
        e_p = 0.5 * (e + np.abs(e))
        b += 2 * (yita) * e_p
        a = np.matmul(np.matmul(np.linalg.inv(np.matmul(w.T, w)), w.T), b)
        k += 1

        if any(e) < 0 and any(e) > 0: break
        if all(np.abs(e) <= th_b): return a, e
    
    print ("No solution found !")
    return None, None

def show_result(w1, w2, a):
    """
    可视化结果
    Parameters:
        w1: 类1样本点
        w2: 类2样本点
        a:  权向量
    """
    import matplotlib.pyplot as plt
    # 可视化类1样本点
    w_1 = np.array(w1)
    x = w_1[:, 0]
    y = w_1[:, 1]
    plt.scatter(x, y, marker = '.',color = 'red')
    # 可视化类2样本点
    w_2 = np.array(w2)
    x = w_2[:, 0]
    y = w_2[:, 1]
    plt.scatter(x, y, marker = '.',color = 'blue')
    # 可视化判别面
    if a != None:
        x = np.arange(-10, 10, 0.1)
        y = -a[0]/a[1]*x - a[2]/a[1]
        plt.plot(x, y)

    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('Classfication Result')
    plt.show()
    

if __name__ == "__main__":
    w_1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
    w_2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
    w_3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
    w_4 = [[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]]
    # (a)
    a, b = HK_algorithm(w_1, w_3)
    print (a, b)
    show_result(w_1, w_3, a)
    # (b)
    a, b = HK_algorithm(w_2, w_4)
    print (a, b)
    show_result(w_2, w_4, a)

