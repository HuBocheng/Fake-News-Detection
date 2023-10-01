import numpy as np
import pandas as pd
import jieba
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer  # 文本特征抽取
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv("./train.news.csv")
data2 = pd.read_csv("./test.news.csv")


def cutword(text):
    return " ".join(list(jieba.cut(text)))


def KNN_demo():
    return None


def bayes_demo():
    # 选取可用特征
    x_train = data.iloc[:, [0, 1, 4]]
    y_train = np.array(data.iloc[:, 5])

    x_test = data2.iloc[:, [0, 1, 4]]
    y_test = data2.iloc[:, 5]

    # 评论分条
    x_train["Report Content"]=x_train["Report Content"].apply(lambda x:x.split("##"))
    x_test["Report Content"]=x_test["Report Content"].apply(lambda x:x.split("##"))

    # 做中文分词

    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["new"].apply(cutword)
    x_train = x_train["new"].tolist()

    t = pd.DataFrame(x_test.astype(str))
    x_test["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    x_test = x_test.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    t = pd.DataFrame(x_test.astype(str))
    x_test["new"] = t["new"].apply(cutword)
    x_test = x_test["new"].tolist()

    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # x_train=x_train.applymap(func)
    # x_test = x_test.applymap(func2)
    # x_train=x_train[:,0]

    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)

    # 5）模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接必读真实值和预测值：\n", y_test == y_predict)  # 直接比对

    print('分类报告：\n', classification_report(y_test, y_predict))

    FPR, TPR, threshold = roc_curve(y_test, y_predict, pos_label=1)
    AUC = auc(FPR, TPR)
    print("AUC:", AUC)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：", score)

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
    return None


if __name__ == "__main__":
    bayes_demo()
    print(123)
