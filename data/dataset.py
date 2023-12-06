from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import AutoTokenizer
import json


def load_SubtaskA_data(num_samples):
    with open('data/SubtaskA/subtaskA_train_monolingual.jsonl', 'r') as f:
        train = [json.loads(line) for line in f.readlines()][0:num_samples]
    with open('data/SubtaskA/subtaskA_dev_monolingual.jsonl', 'r') as f:
        test = [json.loads(line) for line in f.readlines()][0:num_samples]
    return train, test


class ClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {"text": self.texts[item], "label": self.labels[item]}


class ClassificationCollator(object):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __call__(self, batch):
        input_texts = []
        for ex in batch:
            input_text = ex['text']
            input_texts.append(input_text)
        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_len,
            return_tensors='pt',
            padding=True,
            truncation=True,
            return_offsets_mapping=False)
        labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.float)

        return inputs, labels
