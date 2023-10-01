# Fake-News-Detection
NKU_2022Fall Python language programming project

**虚假新闻检测**项目，简单的**nlp分类**问题

使用**机器学习**、**深度学习**和**bert模型**解决问题

仓库中只上传了代码，大文件统统没有上传，下面网盘链接用于下载需要的大文件，照着文件夹融合网盘文件和代码文件即可

[所需附件下载链接](https://pan.baidu.com/s/1WpDSuQgC1HQaVNc8xlpuyQ?pwd=jzkl )

### 问题描述

数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在train.news.csv，测试数据保存在test.news.csv。 实验过程中先统计分析训练数据【train.news.csv】。根据train.news.csv中的Title文字训练模型，然后在test.news.csv上测试，给出Precision, Recall, F1-Score, AUC的结果。

### 环境配置

使用anaconda集成开发环境，pytorch深度学习框架

具体配置方法我参考的博客链接：[PyTorch环境配置及安装_pytorch配置-CSDN博客](https://blog.csdn.net/weixin_43507693/article/details/109015177)

### 方法介绍

#### 机器学习模型

主要流程就是数据加载、预处理、特征工程、模型训练与评估，nlp的任务需要将文本数据转换成向量数据，这里用了词袋模型和`tyidf`两张方法。

代码在`traditional.py`中，都有现成的包用，简单调包调参就行，使用了随机森林、支持向量机、朴素贝叶斯、逻辑回归等方法，有的算法可以加入网格搜索与交叉验证调参，不过感觉如果想继续优化可能得在特征工程部分下手。

最后得到的结果：

|             使用模型             | 向量化方法 |  acc   | recall(1) | precision（1） | auc  |
| :------------------------------: | :--------: | :----: | :-------: | :------------: | :--: |
|     朴素贝叶斯+jieba精确模式     |  词袋模型  | 84.33% |   0.60    |      0.47      | 0.74 |
|               同上               |   tyidf    | 88.97% |   0.33    |      0.80      | 0.66 |
| 高斯内核支持向量机+jieba搜索引擎 |  词袋模型  | 86.62% |   0.10    |      0.84      | 0.55 |
|               同上               |   tyidf    | 91.21% |   0.46    |      0.89      | 0.72 |
|      随机森林+jieba精确模式      |  词袋模型  | 87.03% |   0.12    |      0.97      | 0.56 |
|               同上               |   tyidf    | 87.18% |   0.13    |      0.98      | 0.56 |
|      逻辑回归+jieba精确模式      |  词袋模型  | 90.48% |   0.50    |      0.77      | 0.74 |
|               同上               |   tyidf    | 89.33% |   0.37    |      0.79      | 0.68 |

#### 神经网络解决

一整套神经网络训练框架，`models`文件夹下是神经网络的相关配置参数与神经网络层次信息，`Dataset`文件夹下是数据集与预训练好的`word2vec`向量表还有停用词等一些辅助文件，`saved_dict`是模型训练结果，项目入口在`run.py`中，`utils.py`和`train_eval.py`分别是数据处理函数与模型训练测试函数的打包，详情见下面的directory tree

```bash
structure
│  run.py  程序入口
│  train_eval.py  模型训练和测试的函数
│  utils.py  数据处理的函数
│
├─models
│  │  CNN.py  CNN模型类、Config设置类的定义
│  │  RNN.py  CNN模型类、Config设置类的定义
│  │
│  └─__pycache__  缓存文件
│          CNN.cpython-36.pyc
│          RNN.cpython-36.pyc
│
├─Dataset
│  ├─data
│  │      class.txt  类别名称
|  |      test.news.csv  训练集
|  |      train.news.csv  测试集
│  │      embedding_SougouNews_plus.npz  从词向量库中筛选出来的本项目需要用到的向量
│  │      sgns.sogounews.bigram-char  github上项目Chinese Word Vectors 中文词向量，下载的经过预训练的词向量库
│  │      vocab.pkl  词表
│  │      stop_words.txt  停用词表
│  │
│  ├─log
│  │
│  │
│  └─saved_dict  模型训练结果
│          model.ckpt
│          CNN.ckpt
│          RNN.ckpt
│
└─__pycache__  缓存文件
        train_eval.cpython-36.pyc
        utils.cpython-36.pyc


```

**代码执行流**：

1. 源数据导入，特征筛选
2. 选择神经网络类别，导入相应配置（配置参数和网络架构在`models`文件夹里）
3. 构建训练集 验证集 测试集数据（文字转数据）及其迭代器；构建词表
4. 神经网络初始化和训练模型

**代码解析：**

**utils.py：**

- **函数 `cutword`：**对输入文本进行分词，并移除停用词。用的是`jieba`分词工具。

- **函数 `fun1`**:列表转字符串
- **函数 `build_vocab`**:构建词汇表
- **函数 `build_dataset`**:构建数据集（可以选择使用分词还是字符作为特征）
- **`DatasetIterater`类**：为深度学习模型提供批量的训练或测试数据，将大数据集分成小批次，保存`batch_size`、`device`、`n_batches`等数据，重写了`__next__` 方法，逐个返回下一批数据，全部返回完会抛异常。此外还定义了一系列辅助方法，比如数据转换为 PyTorch 的张量、数据当前批次数等等
- **函数 `build_iterator`**:使用`DatasetIterater`类为数据集构建迭代器。

**train_eval.py：**

- **神经网络初始化函数`init_network`：**对模型的权重进行初始化。初始化方法可以是 '`xavier`'、'`kaiming`' 等。，还可以设定不初始化的部分
- **神经网络训练函数`train`：**模型训练，传入模型配置信息和模型本身，还有训练集测试集验证数据迭代器进行训练，使用的是交叉熵损失函数（针对分类问题），总共迭代epoch次，对于每一轮的每一批数据通过前向传播计算模型输出，接着计算损失函数取值，反向传播计算梯度，使用优化器更新模型参数。每40N批验证一次，打印输出结果。此外还设置了提前终止和日志记录，很完备。
- **神经网络测试函数`test`：**模型测试，传入模型配置信息和模型本身，还有测试集数据迭代器，加载模型权重使用`evaluate`函数评估模型并打印输出
- **神经网络评估函数`evaluate`：**对于 `data_iter` 中的每批数据，依旧是前向传播、计算损失函数，形成预测标签，真实标签和预测标签对比，根据`test`参数选择是否计算分类报告、输出混淆矩阵、计算AUC



总体流程和机器学习一样，只不过文本数据向量化的方法使用的是预训练好的`word2vec`向量，将预训练的向量表导入作为神经网络的`embedding`层，结合全连接层、卷积层、`dropout`层等等构建网络，神经网络配置参数（比如随机失活、提前结束训练batch数目、epoch数目、学习率、数据集位置等等）`models`文件夹下相应py文件的`Config`类里。

要先在`utils.py`中运行主函数生成词表字典，保存为`vocab.pkl`，并在总的预训练300维向量表中抽取我们要用到的向量，保存为`embedding_SougouNews_plus`，后续运行`run.py`，其中要加载上面词表字典和300维预训练向量



#### bert模型调入

将`bert`模型作为一个`model`集成到了神经网络的训练框架中，新增文件夹`bert_pretrain` ，其中`pytorch_model.bin` 文件通常存储bert模型的参数，`bert_config.json` 文件存储了模型的配置，如隐藏层大小、层数、头数等，`vocab.txt` 文件存储了词汇表

详情见下面的directory tree

```bash
structure
│  run.py  程序入口
│  train_eval.py  模型训练和测试的函数
│  utils.py  数据处理的函数
│
├─models
│  │  bert.py
│  │
│  └─__pycache__  缓存文件
│          CNN.cpython-36.pyc
│          RNN.cpython-36.pyc
│
├─Dataset
│  ├─data
│  │      class.txt  类别名称
|  |       test.news.csv  训练集
|  |       train.news.csv  测试集
│  │
│  │
│  │
│  └─saved_dict  模型训练结果
│          model.ckpt
│          bert.ckpt
│
│
├─bert_pretrain  bert参数、模型本身、词表
│        bert_config.json
│        pytorch_model.bin
│        vocab.txt
│
│
├─ERNIE_pretrain
│        bert_config.json
│        pytorch_model.bin
│        vocab.txt
│
│
│
├─pytorch_pretrained
│        ......（一堆.py）
│
│
└─__pycache__  缓存文件
        train_eval.cpython-36.pyc
        utils.cpython-36.pyc
```

总体架构就是这样，不过本机的GPU太烂，跑不了bert这种大语言模型（光参数就几百万上亿......），后面从**colab**上白嫖了一个GPU完成的本次任务，比较迷的就是colab上没法从`pytorch_pretrained`中`import BertModel, BertTokenizer`，虽然确实新版是在`transformers` 库中实现的 类`BertModel, BertTokenizer`，但colab怎么也弄不上去，最后只能找见`pytorch_pretrained`的源代码（都在同名文件夹下），手动在colab上导入各种辅助的类和函数，弄了一堆代码块。

详情就看**`comeon.ipynb`**文件里的东西哇，核心的函数变动很小（就`run.py`、`train_eval.py` 、 `utils.py`中的代码）加了一些bert特性的东西，大多是因为导不进来包手动加的代码块，**代码运行结果在这个ipynb文件里了**（输出不完全的展看开就全有了），**传colab上应该能直接跑**（当然你得把文件路径设定好，该上传的bert配置文件和源数据上传上去，就bert_pretrain和data文件夹里的）

最后最好的acc有92%多，不过参数没记，也没多调参数，没多做特征工程，上限差不多是这里了。应该还是特征工程可以提高上限，除此以外还发现这个**源数据其实是不平衡的**（大概5:1吧），可能得加一些偏置操作，结合过采样欠采样、不平衡学习库等办法或许会好些

用bert模型解决这个问题属于是半作弊的方法，谷歌搞出来的这个大模型在各种NLP任务上都表现很好，其实直接调用了人家的成果做我们的demo，bert的详细原理和特性可以参考下面的链接，包括其独有的多头注意力机制、双向性加持的MLM(Masked Language Model)、NSP (Next Sentence Prediction)机制，这些在当时都是革命性的。

### 参考链接

参考的相关博客与仓库（最主要参考的是第一个仓库，项目条理很清楚）：

[Chinese-Text-Classification-Pytorch: 中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention，DPCNN，Transformer，基于pytorch，开箱即用。](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

[Embedding/Chinese-Word-Vectors: 100+ Chinese Word Vectors 上百种预训练中文词向量 ](https://github.com/Embedding/Chinese-Word-Vectors)

[jieba分词4种模式示例_jieba分词实例-CSDN博客](https://blog.csdn.net/weixin_40122615/article/details/103920499)

[BERT模型精讲 - 知乎 ](https://zhuanlan.zhihu.com/p/150681502)