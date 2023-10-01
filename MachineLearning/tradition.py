import numpy as np
import pandas as pd
import jieba
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本特征抽取
from sklearn.feature_extraction.text import CountVectorizer  # 词袋模型文本特征抽取
from sklearn.naive_bayes import MultinomialNB  # 贝叶斯估计器
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # 决策树估计器与可视化
from sklearn.ensemble import RandomForestClassifier  # 随机森林估计器
from sklearn.model_selection import GridSearchCV  # 网格搜索与交叉验证
from sklearn.svm import SVC  # 支持向量机
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import classification_report  # 分类报告
from sklearn.metrics import roc_curve, auc  # 计算AUC，绘制ROC曲线
import matplotlib.pyplot as plt

# 数据导入 停用词导入
data = pd.read_csv("./train.news.csv")
data2 = pd.read_csv("./test.news.csv")
stopwords = [line.strip() for line in open('./stop_words.txt', 'r', encoding='utf-8').readlines()]


def cutword(text):  # 去停用词的分词函数
    init_cutwordlist = list(jieba.cut(text))
    final_cutword = " "
    for word in init_cutwordlist:
        if word not in stopwords:
            final_cutword += word + " "
    return final_cutword


def cutword2(text):  # 不去停用词的分词函数
    return " ".join(list(jieba.cut(text,use_paddle=True)))  # 参数修改可选用四种分词模式


def fun1(x):
    return " ".join(x)


def dataoperaion(x_train, x_test):  # 数据预处理
    # 评论分条
    x_train["Report Content"] = x_train["Report Content"].apply(lambda x: x.split("##"))
    x_train["Report Content"] = x_train["Report Content"].apply(fun1)
    x_test["Report Content"] = x_test["Report Content"].apply(lambda x: x.split("##"))
    x_test["Report Content"] = x_test["Report Content"].apply(fun1)

    # 做中文分词并整合特征
    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["new"].apply(cutword2)  # 可选用去停用词和不去停用词的分词函数
    x_train = x_train["new"].tolist()

    t = pd.DataFrame(x_test.astype(str))
    x_test["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    x_test = x_test.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    t = pd.DataFrame(x_test.astype(str))
    x_test["new"] = t["new"].apply(cutword2)  # 可选用去停用词和不去停用词的分词函数
    x_test = x_test["new"].tolist()
    return x_train, x_test


def tyidf(x_train, x_test):  # 向量化
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    return x_train, x_test


def countvec(x_train, x_test):  # 向量化
    transfer=CountVectorizer(min_df=1, ngram_range=(1,1),stop_words=stopwords)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    return x_train, x_test


def model_eva(y_test, y_predict):  # 模型评估
    print("y_predict:\n", y_predict)
    print("直接必读真实值和预测值：\n", y_test == y_predict)  # 直接比对
    print('分类报告：\n', classification_report(y_test, y_predict))
    # 计算AUC
    FPR, TPR, threshold = roc_curve(y_test, y_predict, pos_label=1)
    AUC = auc(FPR, TPR)
    print("AUC:", AUC)
    # ROC曲线绘制
    '''
    plt.figure()
    plt.title('ROC CURVE (AUC={:.2f})'.format(AUC))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot(FPR, TPR, color='g')
    plt.plot([0, 1], [0, 1], color='m', linestyle='--')
    plt.show()
    '''


def bayes_demo():
    # 选取可用特征
    x_train = data.iloc[:, [0, 1, 4]]
    y_train = np.array(data.iloc[:, 5])
    x_test = data2.iloc[:, [0, 1, 4]]
    y_test = data2.iloc[:, 5]
    # 数据处理
    x_train, x_test = dataoperaion(x_train, x_test)
    # 文本向量化（tyidf方法/词袋模型）
    x_train, x_test = tyidf(x_train, x_test)
    # x_train, x_test = countvec(x_train, x_test)
    # 贝叶斯估计器算法
    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 模型评估
    y_predict = estimator.predict(x_test)
    model_eva(y_test, y_predict)
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：", score)


def random_forest():
    # 选取可用特征
    x_train = data.iloc[:, [0, 1, 4]]
    y_train = np.array(data.iloc[:, 5])
    x_test = data2.iloc[:, [0, 1, 4]]
    y_test = data2.iloc[:, 5]
    # 数据处理
    x_train, x_test = dataoperaion(x_train, x_test)
    # 文本向量化（tyidf方法/词袋模型）
    # x_train, x_test = tyidf(x_train, x_test)
    x_train, x_test = countvec(x_train, x_test)

    # 随机森林分类算法
    estimator = RandomForestClassifier()
    # 网格搜索与交叉验证
    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    # 调用算法
    estimator.fit(x_train, y_train)
    # 模型评估
    y_predict = estimator.predict(x_test)
    model_eva(y_test, y_predict)
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：", score)
    # 交叉验证结论
    print("最佳参数：", estimator.best_params_)
    print("最佳结果：", estimator.best_score_)
    print("最佳估计器：", estimator.best_estimator_)


def svm():
    # 选取可用特征
    x_train = data.iloc[:, [0, 1, 4]]
    y_train = np.array(data.iloc[:, 5])
    x_test = data2.iloc[:, [0, 1, 4]]
    y_test = data2.iloc[:, 5]
    # 数据处理
    x_train, x_test = dataoperaion(x_train, x_test)
    # 文本向量化（tyidf方法/词袋模型）
    x_train, x_test = tyidf(x_train, x_test)
    # x_train, x_test = countvec(x_train, x_test)
    # 支持向量机分类算法
    estimator = SVC(kernel='rbf')
    # 调用算法
    estimator.fit(x_train, y_train)
    # 模型评估
    y_predict = estimator.predict(x_test)
    model_eva(y_test, y_predict)
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：", score)


def logical():
    # 选取可用特征
    x_train = data.iloc[:, [0, 1, 4]]
    y_train = np.array(data.iloc[:, 5])
    x_test = data2.iloc[:, [0, 1, 4]]
    y_test = data2.iloc[:, 5]
    # 数据处理
    x_train, x_test = dataoperaion(x_train, x_test)
    # 文本向量化（tyidf方法/词袋模型）
    #x_train, x_test = tyidf(x_train, x_test)
    x_train, x_test = countvec(x_train, x_test)
    # 逻辑回归分类算法
    estimator = LogisticRegression()
    # 调用算法
    estimator.fit(x_train, y_train)
    # 模型评估
    y_predict = estimator.predict(x_test)
    model_eva(y_test, y_predict)
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：", score)


if __name__ == "__main__":
    # bayes_demo()
    # random_forest()
    svm()
    # logical()
    print(123)
