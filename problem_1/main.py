"""
@leofansq
https://github.com/leofansq
"""
import numpy as np
import matplotlib.pyplot as plt

def normfunc(x,mu,sigma):
    """
    Return a list of value depending on the parameters according to Normal distribution.
    Parameters:
        x: a list of x value
        mu: mean value
        sigma: std
    """
    y = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return y

def gen_cal_plot(N):
    """
    i) Generate N samples and plot them.
    ii) Calculate the mean & std and plot.
    Parameters:
        N : the number of the samples
    """
    S = []
    # 当剩余样本数大于1000时随机确定此轮采样样本个数n
    while N >= 1000:
        n = np.random.randint(1,1001)
        N -= n
        # 随机选取此轮采样上下界
        x_l, x_u = np.random.randint(-100, 101, 2)
        if x_l > x_u: x_l, x_u = x_u, x_l
        # 采样
        s = np.array(np.random.uniform(x_l, x_u+1, n), np.int)
        S.extend(s)
    # 当剩余样本数小于1000时，完成最后一轮采样
    if N:
        x_l, x_u = np.random.randint(-100, 101, 2)
        if x_l > x_u: x_l, x_u = x_u, x_l
        s = np.array(np.random.uniform(x_l, x_u+1, N), np.int)
        S.extend(s)
    # 计算均值和方差
    mean = np.mean(S)
    std = np.std(S)
    # 根据均值方差计算正态分布曲线
    x = np.arange(-100, 101, 1)
    y = normfunc(x, mean, std)
    # 绘图
    plt.hist(S, bins=200, range=[-100,100], normed=True, color='orange')
    plt.plot(x, y)
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    gen_cal_plot(10000)
    gen_cal_plot(100000)
    gen_cal_plot(1000000)