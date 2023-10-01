# coding: UTF-8
import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import jieba
from pandas.core.frame import DataFrame

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
stopwords = [line.strip() for line in open('./Dataset/data/stop_words.txt', 'r', encoding='utf-8').readlines()]


def cutword(text):  # 去停用词的分词函数
    init_cutwordlist = list(jieba.cut(text))
    final_cutword = " "
    for word in init_cutwordlist:
        if word not in stopwords and word != ' ':
            final_cutword += word + " "
    return final_cutword


def fun1(x):
    return " ".join(x)


def build_vocab(x_train, tokenizer, max_size, min_freq):
    vocab_dic = {}
    list1 = np.array(x_train["new"])
    for line in list1:
        content = line  # content放文本
        for word in tokenizer(content):  # 遍历分词器处理后的content，给vocab_dic加东西
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(x_train, y_train, x_dev, y_dev, x_test, y_test, config, ues_word):
    if ues_word:  # 使用分词工具得到的分词作为特征
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        t = pd.DataFrame(x_train.astype(str))
        x_train["new"] = t["new"].apply(cutword)
        t = pd.DataFrame(x_dev.astype(str))
        x_dev["new"] = t["new"].apply(cutword)
        t = pd.DataFrame(x_test.astype(str))
        x_test["new"] = t["new"].apply(cutword)

    else:  # 使用单个字符作为特征
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(x_train, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(x, y, pad_size=32):
        contents = []
        list1 = np.array(x)
        list2 = np.array(y)
        for i in range(list1.shape[0]):
            content, label = list1[i][0], list2[i]  # content放文本，label放0/1
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(x_train, y_train, config.pad_size)
    dev = load_dataset(x_dev, y_dev, config.pad_size)
    test = load_dataset(x_test, y_test, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size  # 每批元素数
        self.batches = batches  # 数据
        self.n_batches = len(batches) // batch_size  # 分几批
        self.residue = False  # 记录batch数量是否不为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    # 数据导入
    data = pd.read_csv("./Dataset/data/train.news.csv")
    data["Report Content"] = data["Report Content"].apply(lambda x: x.split("##"))
    data["Report Content"] = data["Report Content"].apply(fun1)
    x_train = data.iloc[:, [0, 1, 4]]
    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["Title"] + ' ' + t["Ofiicial Account Name"] + ' ' + t["Report Content"]
    x_train = x_train.drop(["Title", "Ofiicial Account Name", "Report Content"], axis=1)
    t = pd.DataFrame(x_train.astype(str))
    x_train["new"] = t["new"].apply(cutword)
    # 构建词表并提取预训练词向量（暴力倒入all词向量库会爆内存）
    vocab_dir = "./Dataset/data/vocab.pkl"
    pretrain_dir = "./Dataset/data/sgns.sogounews.bigram-char"
    emb_dim = 300 # 词向量维度固定300
    filename_trimmed_dir = "./Dataset/data/embedding_SougouNews_plus"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(x_train, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
