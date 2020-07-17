# Problem 5
## 1. 问题描述

请写一个程序,实现 MSE 多类扩展方法。每一类用前 8 个样本来构造分类器,用后两个样本作测试。请给出你的正确率.

<div align=center>
    <img src="./data.png" width='400'>
</div>

## 2. 实现思路

* 根据样本类别建立Label矩阵Y
* 基于MSE多类扩展方法计算权向量$a = W^+ Y^T$

## 3. Python代码
### 3.1 MSE 多类扩展方法
```Python
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
```

### 3.2 测试函数,计算正确率
```Python
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
```

### 3.3 求解
```Python
w_1 = [[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8], 
      [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]]
w_2 = [[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9], 
      [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]]
w_3 = [[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], 
      [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]]
w_4 = [[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], 
      [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]]

wi = [w_1[:8], w_2[:8], w_3[:8], w_4[:8]]
a = MSE_multi(wi)
print (a)

w_test = [w_1[8:], w_2[8:], w_3[8:], w_4[8:]]
f_ratio = test(w_test, a)
print (f_ratio)
```

## 4. 结果与讨论

基于MSE多类扩展方法,利用每一类的前8个样本,计算得到权向量a为

$$\left[
 \begin{matrix}
   0.02049668 & 0.06810971 & -0.04087307 & -0.04773332 \\
   0.01626151 & -0.03603827 & 0.05969134 & -0.03991458 \\
   0.26747287 & 0.27372075 & 0.25027714 & 0.20852924
  \end{matrix}
\right]$$

经测试, 上述权向量可以将测试样本全部正确分类,错误率为0.


