import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jieba

from sklearn.datasets import load_iris  # 加载数据集

from sklearn.model_selection import train_test_split  # 数据集划分

from sklearn.feature_extraction import DictVectorizer  # 特征抽取之字典特征抽取
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # 特征抽取之文本特征抽取

from sklearn.feature_selection import VarianceThreshold  # 低方差特征过滤
from scipy.stats import pearsonr  #计算方差
from  sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 数据预处理

'''
data_train = pd.read_csv('./iris_training.csv', encoding='gbk')  # 训练集
data_test = pd.read_csv('./iris_test.csv', encoding='gbk')  # 测试集

print("训练集\n",data_train)
print("测试集\n",data_test)
#训练集
new_train = data_train[['花瓣长度', '花瓣宽度', '花萼长度', '花萼宽度']].values  # 鸢尾花训练集特征值(4个)
new_train.shape
iris_target = data_train['类别'].values
iris_target.shape

#测试集
new_test = data_test[['花瓣长度', '花瓣宽度', '花萼长度', '花萼宽度']].values  # 鸢尾花测试集特征值(4个)
new_test.shape
iris_target_test = data_test['类别'].values
iris_target_test.shape
'''


def load_demo():
    iris = load_iris()
    # print("鸢尾花数据集：\n", iris)
    # print("查看数据集描述：\n", iris["DESCR"])  # 数据集的描述信息
    # print("查看特征值的名字：\n", iris.feature_names)
    # print("查看特征值：\n", iris.data, iris.data.shape)  # shape:(150,4)
    # print("查看标签",iris["target_names"])
    # print(iris.target)
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)
    print("训练集的目标值：\n",y_train,type(y_train))
    return None


# 特征工程


def dic_demo():
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    data2 = ['life is short,i like like python',
             'life is too long,i dislike python']
    # 一、字典特征抽取
    # 1、实例化一个类(转化器类）
    transfer = DictVectorizer(sparse=False)  # 完整矩阵表示
    transfer2 = DictVectorizer()  # 稀疏矩阵表示（节省空间）
    # 2、用类中的成员函数fit_transform
    data_new = transfer.fit_transform(data)
    data_new2 = transfer2.fit_transform(data)
    print(data_new)
    print(data_new2)
    # /3、矩阵各行意义（特征名字）
    print(transfer.get_feature_names_out())  # get_feature_names_out()函数是转换器事例的成员函数，
    # 得先fit_transform（）后加载完数据才能调用
    return None


def count_demo():
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    data2 = ['life is short,i like like python',
             'life is too long,i dislike python']
    #  二、文本特征抽取
    # 1、实例化一个类(转化器类）
    transfer = CountVectorizer()  # 完整矩阵表示
    # 2、用类中的成员函数fit_transform
    data_new = transfer.fit_transform(data2)  # 返回的是一个matrix类型的，有成员函数将其从稀疏矩阵变成标准矩阵
    print(data_new.toarray(), type(data_new))
    print(transfer.get_feature_names_out())
    return None


def cutword(text):
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo():
    data2 = ['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
             '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
             '如果只用一种方式了解某件事物，他就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']
    datacut = []
    for i in data2:
        datacut.append(cutword(i))
    print(datacut)
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(datacut)
    print(data_new.toarray())
    print(transfer.get_feature_names_out())
    return None


def tfidf():
    data2 = ['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
             '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
             '如果只用一种方式了解某件事物，他就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']
    datacut = []
    for i in data2:
        datacut.append(cutword(i))
    print(datacut)

    transfer = TfidfVectorizer()
    data_new = transfer.fit_transform(datacut)
    print(data_new.toarray())
    print(transfer.get_feature_names_out())
    return None


def minmax():  # 归一化
    data = pd.read_csv("C:/Users/Thinkbook/Desktop/dateset.csv")
    data = data.iloc[:, 0:3]
    # print(data)
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler(feature_range=[1, 2])  # 可设定归一后的范围
    data_new = scaler2.fit_transform(data)
    print(data_new)
    return None


def standerd():  # 标准化
    data = pd.read_csv("C:/Users/Thinkbook/Desktop/dateset.csv")
    data = data.iloc[:, 0:3]
    # print(data)
    scaler = StandardScaler()
    data_new = scaler.fit_transform(data)
    print(data_new)
    return None

def variance():
    """
        低方差特征过滤
        :return:
        """
    # 1、获取数据
    data = pd.read_csv("C:/Users/Thinkbook/Desktop/dateset.csv")
    print('data:\n', data)
    data = data.iloc[:, 1:-2]
    print('data:\n', data)

    # 2、实例化一个转换器类
    # transform = VarianceThreshold()
    transform = VarianceThreshold(threshold=10) # 设定方差小于等于多少的的特征被过滤，默认0

    # 3、调用fit_transform
    data_new = transform.fit_transform(data)
    print("data_new\n", data_new, data_new.shape)

    r = pearsonr(data["pe_ratio"], data["pb_ratio"])
    return  None

def pca():
    data=[[4,5,5,6],[8,7,9,2]]

    transfer=PCA(n_components=2)
    data_new=transfer.fit_transform(data)
    print("data_new\n",data_new)

    transform2 = PCA(n_components=0.95)  # 保留95%的信息
    data_new2 = transform2.fit_transform(data)
    print("data_new2\n", data_new2)

    return None

if __name__ == "__main__":
    # 用的transfer
    # dic_demo()
    # count_demo()
    tfidf()
    # 用的scaler
    # minmax()
    # standerd()
    # pca()
    # load_demo()
