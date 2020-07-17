"""
@leofansq
https://github.com/leofansq
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 设置参数
    mean_1 = [1, 0]
    mean_2 = [-1, 0]
    cov = np.eye(2)

    # 生成样本点
    s_1 = np.random.multivariate_normal(mean_1, cov, 50)
    s_2 = np.random.multivariate_normal(mean_2, cov, 50)
    
    # 绘制样本点
    x_1 = s_1[:,0]
    y_1 = s_1[:,1]
    x_2 = s_2[:,0]
    y_2 = s_2[:,1]
    plt.subplot(211).set_title("Real Distribution")
    plt.scatter(x_1,y_1)
    plt.scatter(x_2,y_2)

    # 根据贝叶斯判决边界绘制判决结果
    S = []
    S.extend(s_1)
    S.extend(s_2)
    X_1 = []
    Y_1 = []
    X_2 = []
    Y_2 = []
    for i in S:
        if i[0]>0:
            X_1.append(i[0])
            Y_1.append(i[1])
        else:
            X_2.append(i[0])
            Y_2.append(i[1])
    plt.subplot(212).set_title("Decision Distribution")
    plt.scatter(X_1,Y_1)
    plt.scatter(X_2,Y_2)
    plt.tight_layout()
    plt.show()

    # 计算错误率
    err_1 = [i for i in s_1 if i[0]<0]
    err_2 = [i for i in s_2 if i[0]>0]
    err_cnt = len(err_1) + len(err_2)
    rate = err_cnt/100
    print (rate)


if __name__ == "__main__":
    main()

    


