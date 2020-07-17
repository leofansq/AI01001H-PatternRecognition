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

def batch_perception(w1, w2):
    """
    Batch Perception
    Parameters:
        w1: 类1样本
        w2: 类2样本
    Return:
        a
        k: 迭代次数
    """
    # 规范化增广样本
    w = samples_trans(w1, w2)

    # 初始化参数
    a = np.zeros_like(w[1])
    yita = 1
    theta = np.zeros_like(w[1])+1e-6
    k = 0

    # 迭代
    while True:
        y = np.zeros_like(w[1])
        for i in w:
            if np.matmul(a.T, i) <= 0:y += i
        yita_y = yita * y

        if all(np.abs(yita_y)<=theta):break

        a += yita_y
        k += 1
    
    return a, k

def show_result(w1, w2, a):
    import matplotlib.pyplot as plt

    w_1 = np.array(w1)
    x = w_1[:, 0]
    y = w_1[:, 1]
    plt.scatter(x, y, marker = '.',color = 'red')

    w_2 = np.array(w2)
    x = w_2[:, 0]
    y = w_2[:, 1]
    plt.scatter(x, y, marker = '.',color = 'blue')

    x = np.arange(-10, 10, 0.1)
    y = -a[0]/a[1]*x - a[2]/a[1]
    plt.plot(x, y)
    
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('Classfication Result')
    plt.show()




if __name__ == "__main__":
    # (a)
    w_1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
    w_2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
    a, k = batch_perception(w_1, w_2)
    print ("a:{}\nk:{}".format(a, k))
    show_result(w_1, w_2, a)
    # (b)
    w_3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
    a, k = batch_perception(w_3, w_2)
    print ("a:{}\nk:{}".format(a, k))
    show_result(w_3, w_2, a)
