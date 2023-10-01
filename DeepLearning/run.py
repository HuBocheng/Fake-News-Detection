# coding: UTF-8
import time
import torch
import numpy as np
import pandas as pd
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif
from sklearn.model_selection import train_test_split
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="RNN")
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
args = parser.parse_args()


def fun1(x):
    return " ".join(x)


if __name__ == '__main__':
    # 数据导入 特征筛选
    data = pd.read_csv("./Dataset/data/train.news.csv")
    data2 = pd.read_csv("./Dataset/data/test.news.csv")
    data["Report Content"] = data["Report Content"].apply(lambda x: x.split("##"))
    data["Report Content"] = data["Report Content"].apply(fun1)
    data2["Report Content"] = data2["Report Content"].apply(lambda x: x.split("##"))
    data2["Report Content"] = data2["Report Content"].apply(fun1)

    train_set, dev_set = train_test_split(data, test_size=0.2, random_state=24)
    x_train = train_set.iloc[:, [0, 1, 4]]
    y_train = train_set.iloc[:, 5]
    x_dev = dev_set.iloc[:, [0, 1, 4]]
    y_dev = dev_set.iloc[:, 5]
    x_test = data2.iloc[:, [0, 1, 4]]
    y_test = data2.iloc[:, 5]

    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["Title"]  + ' ' + t["Report Content"]
    x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)

    t = pd.DataFrame(x_dev.astype(str))
    x_dev["new"] = t["Title"] + ' ' + t["Report Content"]
    x_dev = x_dev.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)

    t = pd.DataFrame(x_test.astype(str))
    x_test["new"] = t["Title"]  + ' ' + t["Report Content"]
    x_test = x_test.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)


    # 参数传入“设置类”
    dataset = 'Dataset'  # 数据集
    embedding = 'embedding_SougouNews_plus.npz'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


    # 构建训练集 验证集 测试集数据（文字转数据）及其迭代器；构建词表
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(x_train, y_train, x_dev, y_dev, x_test, y_test, config,
                                                           args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 神经网络初始化和训练模型
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
