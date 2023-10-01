import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from datasets import load_dataset

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='data', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


dataset = Dataset('train')
print(len(dataset), dataset[0])


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=10,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break

print(len(loader))
print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)
