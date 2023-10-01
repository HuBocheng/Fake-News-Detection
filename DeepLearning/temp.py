import time
import torch
import numpy as np
import pandas as pd
import jieba
from sklearn.metrics import classification_report

from train_eval import train, init_network
from importlib import import_module
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame


def cutword(text):  # 去停用词的分词函数
    init_cutwordlist = list(jieba.cut(text))
    final_cutword = " "
    for word in init_cutwordlist:
        if word not in stopwords and word!=' ':
            final_cutword += word + " "
    return final_cutword


def fun1(x):
    return " ".join(x)


if __name__ == '__main__':
    # stopwords = [line.strip() for line in open('./THUCNews/data/stop_words.txt', 'r', encoding='utf-8').readlines()]
    # data = pd.read_csv("./train.news.csv")
    # data2 = pd.read_csv("./test.news.csv")
    # data["Report Content"] = data["Report Content"].apply(lambda x: x.split("##"))
    # data["Report Content"] = data["Report Content"].apply(fun1)
    # data2["Report Content"] = data2["Report Content"].apply(lambda x: x.split("##"))
    # data2["Report Content"] = data2["Report Content"].apply(fun1)

    # train_set, dev_set = train_test_split(data, test_size=0.2, random_state=42)
    # x_train = train_set.iloc[:, [0, 1, 4]]
    # y_train = train_set.iloc[:, 5]
    # x_dev = dev_set.iloc[:, [0, 1, 4]]
    # y_dev = dev_set.iloc[:, 5]
    # x_test = data2.iloc[:, [0, 1, 4]]
    # y_test = data2.iloc[:, 5]

    # x_train1 = data.iloc[:, [0, 1, 4]]
    #
    # t = pd.DataFrame(x_train1.astype(str))
    # x_train1["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_train1 = x_train1.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    #
    # t = pd.DataFrame(x_train1.astype(str))
    # x_train1["new"] = t["new"].apply(cutword)
    # t = pd.DataFrame(x_train.astype(str))
    # x_train["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    #
    # t = pd.DataFrame(x_dev.astype(str))
    # x_dev["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_dev = x_dev.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    #
    # t = pd.DataFrame(x_test.astype(str))
    # x_test["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    # x_test = x_test.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)

    # 计算、选取合适的pad_size = 100每句话处理成的长度(短填长切)
    # list1 = np.array(x_train["new"])
    # array2 = np.array([len(auto) for auto in list1])
    # count=0
    # for i in range(len(array2)):
    #     if(array2[i]>80):
    #         count+=1
    # a = array2.mean()
    # b = array2.max()
    # c = array2.min()
    # d=np.median(array2)

    # 构建词表
    # tokenizer = lambda x: [y for y in x]
    # vocab_dic = {}
    # list1 = np.array(x_train["new"])
    # for line in list1:
    #     content = line  # content放文本
    #     for word in tokenizer(content):  # 遍历分词器处理后的content，给vocab_dic加东西
    #         vocab_dic[word] = vocab_dic.get(word, 0) + 1
    # vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= 1], key=lambda x: x[1], reverse=True)[:10000]
    # vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    # vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})

    # test_data_y=pd.DataFrame({"0":[0.1,0.2,0.3,0.6]})
    # pres=np.array([auto for auto in test_data_y["0"]])
    # for i in range(len(pres)):
    #     if pres[i]>=0.5:
    #         pres[i]=1
    #     else:
    #         pres[i]=0
    # label=np.array([1,1,0])
    # print('分类报告：\n', classification_report(label, pres))

    f = open(r"D:\codeing\pycoding\DeepLearning\Dataset\data\sgns.sogounews.bigram-char", encoding="UTF-8")
    t=f.readline().split()
    n, dimension = int(t[0]), int(t[1])
    print(dimension)


    chinesewordvec = f.readlines()
    chinesewordvec = [i.split() for i in chinesewordvec]
    vectorsmap = []
    wordtoindex = indextoword = {}
    for i in range(n):
        vectorsmap.append(list(map(float, chinesewordvec[i][len(chinesewordvec[i]) - dimension:])))
        wordtoindex[chinesewordvec[i][0]]=i
        indextoword[i] = chinesewordvec[i][0]
    f.close()
    print(123)
    # arry2=x_train1["new"].tolist()
    # arry3=[]
    # for i in range(len(arry2)):
    #     arry3.append(len(arry2[i].split()))
    arry2=[len(auto.split()) for auto in x_train1["new"].tolist()]
    print(np.median(arry2))   # 14
    print(np.mean(arry2))     # 24
    #print(np.array([len(auto)for auto in x_train1['new']]).mean())
    print(123)
