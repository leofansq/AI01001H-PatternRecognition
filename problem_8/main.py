"""
Ng谱聚类: load_data(file_name) 数据加载; 
generate_graph(data, k, theta) 图构造; 
k_means(data, mu) K-means; 
ng_algo(W, c) 谱聚类

@leofansq
https://github.com/leofansq
"""
import numpy as np 
import matplotlib.pyplot as plt

def load_data(file_name):
    """
    加载数据
    """    
    data = []
    with open(file_name, 'r') as f:
        content = f.readlines()
        for i in content:
            i = i[:-1].split(" ")
            data.append([float(i[0]), float(i[1])])
    data = np.array(data)

    return data

def generate_graph(data, k, theta):
    """
    构造图
    Parameter:
        data: 待聚类数据
        k: k近邻数
        theta: 亲和性参数
    Return:
        w: 亲和性矩阵
    """
    # 构造data行列矩阵(n*n*d)以便后续矩阵运算: data_c每列相同 = data_r每行相同 = 样本数据
    data_c = np.tile(np.expand_dims(data.copy(), axis=1), (1,data.shape[0],1))
    data_r = np.tile(np.expand_dims(data.copy(), axis=0), (data.shape[0],1,1))

    # 生成Dist矩阵
    dist = np.sum((data_c - data_r)**2, axis=-1)

    # 生成W矩阵
    # 初始化w
    w = np.zeros_like(dist)
    for idx_sample, i in enumerate(dist):
        idx = np.arange(0, i.shape[0])
        # 构造 距离-索引 序列, 将距离和索引一一对应
        i_idx = zip(i, idx)
        # 按照距离递增排序
        i_sorted = sorted(i_idx, key=lambda i_idx: i_idx[0])
        # 生成w矩阵: 循环时排除自身距离为0的干扰
        for j in range(1,k+1):
            w[idx_sample, i_sorted[j][1]] = np.exp(-i_sorted[j][0]/(2*(theta**2)))
    # w调整:为保证w为对称矩阵
    w = (w.T + w)/2
    
    return w

def k_means(data, mu):
    """
    K-means聚类
    Parameters:
        data: 待聚类数据(np.array)
        mu: 初始化聚类中心(np.array)
    Return:
        c: 聚类结果[[第一类数据], [第二类数据], ... , [第c类数据]]
        label: 聚类结果label列表 [样本1类别, 样本2类别, ... , 样本n类别]
        mu: 类中心结果[第一类类中心, 第二类类中心, ... , 第c类类中心]
        cnt: 迭代次数
    """
    # 待聚类数据矩阵调整(复制矩阵使其从n*d变为n*c*d, 便于后续矩阵运算)
    data = np.tile(np.expand_dims(data, axis=1), (1,mu.shape[0],1))
    # 初始化变量
    mu_temp = np.zeros_like(mu) # 保存前一次mu结果
    cnt = 0

    # 迭代更新类中心
    while  np.abs(np.sum((mu - mu_temp)**2))>1e-10 :
        mu_temp = mu
        cnt += 1
        label = np.zeros((data.shape[0]), dtype=np.uint8)
        # mu矩阵调整(复制矩阵使其从c*d变为n*c*d, 便于后续矩阵运算)
        mu = np.tile(np.expand_dims(mu, axis=0), (data.shape[0],1,1))
        # 生成距离矩阵(n*c)
        dist = np.sum((data-mu)**2, axis=-1)
        # 初始化聚类结果 & 根据距离确定样本类别
        c = []
        for _ in range(data.shape[1]):
            c.append([])

        for idx, sample in enumerate(data):
            c[np.argmin(dist[idx])].append(sample[0])
            label[idx] = np.argmin(dist[idx])
        c = np.array(c)
        # 更新类中心
        mu = []
        for i in c: mu.append(np.mean(i, axis=0))
        mu = np.array(mu)

    return c, label, mu, cnt

def ng_algo(W, c):
    """
    Ng谱聚类算法
    Parameters:
        W: 亲和性矩阵
        c: 聚类类别数
    Return:
        label: 聚类结果label列表 [样本1类别, 样本2类别, ... , 样本n类别]
    """
    # 计算D & D^(-1/2)矩阵: 为避免生成D后计算会出现分母为0的情况, 直接计算D^(-1/2)
    W_rowsum = np.sum(W, axis=1)
    D = np.diag(W_rowsum)
    # W_rowsum = 1/(np.sqrt(W_rowsum))
    W_rowsum = W_rowsum**(-0.5)
    D_invsqrt = np.diag(W_rowsum)
    # 计算L矩阵
    L = D - W
    # 计算L_sym矩阵
    L_sym = np.matmul(np.matmul(D_invsqrt, L), D_invsqrt)
    # L_sym特征值 & 特征向量
    e_value, e_vector = np.linalg.eig(L_sym)
    e_vector = e_vector.T
    e = zip(e_value, e_vector)
    e_sorted = sorted(e, key=lambda e: e[0])
    # 生成新特征
    new_feature = []
    for i in range(c):
        new_feature.append(e_sorted[i][1])
    new_feature = np.array(new_feature).T
    # 归一化新特征
    norm_feature = []
    for i in new_feature:
        i = i/(np.sqrt(np.sum(i**2))+1e-10)
        norm_feature.append([i[0], i[1]])
    norm_feature = np.array(norm_feature)
    # 对新特征做K-means
    mu = np.array([norm_feature[50], norm_feature[150]])
    _, label, _, _ = k_means(norm_feature, mu)
    
    return label
    

if __name__ == "__main__":
    # 加载数据 & 可视化
    data = load_data("./data.txt")
    # plt.scatter(data[:,0], data[:,1], marker=".", color="black")
    # plt.title("Data")
    # plt.savefig('data.png')

    # 构造图
    k = 5
    theta = 2
    w = generate_graph(data, k, theta)

    # 谱聚类
    c = 2
    label = ng_algo(w, c)

    # 可视化
    color = ['red', 'blue']
    for idx, i in enumerate(data):
        i = np.array(i)
        plt.scatter(i[0], i[1], marker = '.',color = color[label[idx]])
    plt.title('Result')
    plt.show()

    # # ACC-K/Sigma 曲线
    # gt = np.zeros((200))
    # for i in range(100,200): gt[i]=1

    # Acc = []
    # # for k in range(1,199):
    # for theta in np.arange(0.1,2,0.1):
    #     print (theta)
    #     k = 5
    #     # theta = 1
    #     w = generate_graph(data, k, theta)
    #     c = 2
    #     label = ng_algo(w, c)
    #     acc = (200 - np.sum(np.abs(label - gt)))/200
    #     Acc.append(acc)
    # Acc = np.array(Acc)
    # # plt.plot(np.arange(1,199), Acc)
    # plt.plot(np.arange(0.1,2,0.1), Acc)
    # plt.title("Acc-Sigma (k={})".format(k))
    # plt.xlabel("sigma")
    # plt.ylabel("acc")
    # plt.show()





        
