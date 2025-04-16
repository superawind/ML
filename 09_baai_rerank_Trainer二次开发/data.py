import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets 
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from transformers import AutoTokenizer

# rerank 的数据本质就是将 query + ans 一起进行tokenizer， 并且根据group_size 来确定一个query对应多个样本数据，一个样本是 query + pos 编码，另group_size - 1是 query + neg 编码

from arguments import DataArguments

class TrainDataForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.__dataclass_fields__, file), split='train')
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)

        else:
            print(args.train_data)
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, item):
        query = self.dataset[item]['query']
        pos = random.choice(self.dataset[item]['pos'])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size-1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        batch_data = []
        batch_data.append(self.create_one_sample(query, pos))
        for neg in negs:
            batch_data.append(self.create_one_sample(query, neg))
        
        return batch_data

    def create_one_sample(self, query, ans):
        item = self.tokenizer.encode_plus(
            query,
            ans,
            truncation=True,
            max_length = self.args.max_len,
            padding=False
        )
        return item 
    
@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(self, features):
        if isinstance(features[0], list):
            features = sum(features, [])
        
        return super().__call__(features)

def my_collator(batch):
    # batch 是一个列表，里面包含batchg_size=5个元素，每一个元素都是dataset 中getitem的返回，即，包含group_size 数量的列表
    if isinstance(batch[0], list):
        batch = sum(batch, [])

    return DataCollatorWithPadding(tokenizer=tokenizer).__call__(batch)

tokenizer = AutoTokenizer.from_pretrained('E:/Model/bge-large-zh-v1.5')
if __name__ == '__main__':
    dataset = TrainDataForCE(DataArguments("E:/code/00_demo/FlagEmbedding/FlagEmbedding/reranker/mytest/toy_finetune_data.jsonl", 3, 33), tokenizer=tokenizer)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=5, collate_fn=my_collator)
    for elem in dataloader:
        # elem 是一个tensor 内容{'input_ids':... , 'attention_mask':..., }
        # elem.input_ids.shape = [batch_size* group_size, max_len]
        print(elem)
        break