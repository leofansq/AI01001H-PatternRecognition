"""
SVM训练 & 测试, MNIST数据集

@leofansq
https://github.com/leofansq
"""
import numpy as np
from struct import unpack
from sklearn import svm

import pickle

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

if __name__ == "__main__":
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
    train_svm(train_data, 200, MODEL_PATH)
    # 测试
    test_svm(test_data, MODEL_PATH)
    
