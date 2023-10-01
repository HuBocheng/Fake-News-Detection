from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
print("鸢尾花数据集：\n", iris)
print("查看数据集描述：\n", iris["DESCR"])
print("查看特征值的名字：\n", iris.feature_names)
print("查看特征值：\n", iris.data, iris.data.shape)  # 150个样本

# 数据集划分  X为特征 Y为标签
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
print("训练集的特征值：\n", x_train, x_train.shape)  # 120个样本
