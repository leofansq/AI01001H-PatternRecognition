# Problem 9
## 1. 问题描述

从MNIST数据集中选择两类，对其进行SVM分类，可调用现有的SVM工具.

## 2. 实现思路

为实现上述功能, 需实现以下子功能:

* 加载图像数据与Label数据: Mnist数据集原始图像数据以idx3-ubyte格式存储, Label数据以idx1-ubyte格式存储. 因此需根据其特定格式加载数据.

* 根据需求生成实际使用数据: 由于需从数据集中选取两类进行分类, 因此需生成指定类别的图像数据和其对应的Label数据, 以数组的形式返回.

* SVM训练与模型保存: 利用训练数据对SVM进行训练, 直接调用Sk-learn的SVM工具. 将训练好的模型保存, 以便后续使用.

* SVM测试: 加载已有的SVM模型, 利用测试数据进行测试, 统计并评价分类的准确率.

## 3. Python代码

代码实现时, 使用以下库: numpy, struct.unpack, sklearn.svm, pickle

### 3.1 图像数据与Label数据的加载
```Python
def load_imgs(path):
    """
    加载图像数据
    Parameter:
        path: 图像数据文件路径
    Return:
        imgs: 每行为一个图像数据(n, img.size)
    """
    with open(path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        imgs = np.fromfile(f, dtype=np.uint8).reshape(num, rows*cols)
    return imgs

def load_labels(path):
    """
    加载Label数据
    Parameter:
        path: Label数据文件路径
    Return:
        labels: 每行为一个label标签(n,)
    """
    with open(path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels
```

### 3.2 生成与整合数据
```Python
def generate_data(img_path, label_path, c_list):
    """
    生成整合数据
    Parameter:
        img_path: 图像数据文件路径
        label_path: Label数据文件路径
        c_list: 需生成的类别名列表
    Return:
        [img_c, label_c]: img_c为图像数据数组,每行为一个图像的数据(n,img.size);label_c为标签数据数组,每行为一个标签的数据(n,)
    """
    # 加载图像数据和Label
    imgs = load_imgs(img_path)
    labels = load_labels(label_path)

    # 选取特定类别并生成所需数据
    img_c = []
    label_c = []
    for c in c_list:
        idx = np.where(labels==c)
        img_c.extend(imgs[idx[0]])
        label_c.extend(labels[idx[0]])
    # 图像数据归一化
    img_c = np.array(img_c)/255.0 
    label_c = np.array(label_c)
    
    return [img_c, label_c]
```

### 3.3 SVM训练与模型保存
```Python
def train_svm(train_data, c, model_path):
    """
    SVM训练 & 模型保存
    Parameter:
        train_data: [img_c, label_c], generate_data函数返回的数据格式
        c: SVM参数c
        model_path: 训练生成的模型保存路径
    """
    # SVM训练
    print ("Start Training...")
    classifier = svm.SVC(C=c, decision_function_shape='ovr')
    classifier.fit(train_data[0], train_data[1])

    # 模型保存
    save = pickle.dumps(classifier)
    with open(model_path, 'wb+') as f:  f.write(save)
    print ("Training Done. Model is saved in {}.".format(model_path))
```

### 3.4 SVM测试
```Python
def test_svm(test_data, model_path):
    """
    SVM测试
    Parameter:
        test_data: [img_c, label_c], generate_data函数返回的数据格式
        model_path: 测试的模型的文件路径
    """
    # 加载待测试模型
    print ("Start Testing...")
    with open(model_path, 'rb') as f: s = f.read()
    classifier = pickle.loads(s)

    # 模型测试
    score = classifier.score(test_data[0], test_data[1])
    print ("Testing Accuracy:", score)
```

### 3.5 实验主程序
```Python
# Option
# 文件路径
TRAIN_IMG_PATH = "./data/train-images.idx3-ubyte"
TRAIN_LABEL_PATH = "./data/train-labels.idx1-ubyte"
TEST_IMG_PATH = "./data/t10k-images.idx3-ubyte"
TEST_LABEL_PATH = "./data/t10k-labels.idx1-ubyte"
# 训练&测试的指定类别
CLASS = [1, 2]
# 模型保存路径
MODEL_PATH = "svm.model"

# 数据加载
train_data = generate_data(TRAIN_IMG_PATH, TRAIN_LABEL_PATH, CLASS)
test_data = generate_data(TEST_IMG_PATH, TEST_LABEL_PATH, CLASS)
# 训练
train_svm(train_data, 1, MODEL_PATH)
# 测试
test_svm(test_data, MODEL_PATH)
```

## 4. 结果与讨论

通过运行python main.py利用上述代码对SVM进行训练与模型测试.(为方便作业的上传与下载, 提交时对数据集数据进行了压缩, 测试前需解压)

实验发现, SVM能较好地对数据进行分类. 在参数C相同的情况下, 对于本身相似度较小的两个类别, 如1和2, 测试正确率可达到99.5\%;
对于本身相似度较高的两个类别, 如0和6, 1和7, 测试正确率可达到99.0\%.

在待分类类别固定不变的情况下, 调整参数C可以使测试正确率发生变化. 对于1和2两个类别, 当C=1时, 测试正确率为99.4\%; 当C=100时, 正确率为99.5\%;
当C=200时, 正确率为99.7\%; 但当C=1000时, 正确率降回99.5\%.
上述现象可以从理论上进行分析. 增大C意味着减小对错分情况的松弛, 可以达到更好的分类情况, 但同时也可能带来过拟合的隐患, 一旦引起过拟合, 模型虽然能在训练数据上达到更优的分类效果, 但在测试数据上正确率则会降低.