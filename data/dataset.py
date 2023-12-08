from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import AutoTokenizer
import json
from fastchat.conversation import get_conv_template


def load_SubtaskA_data(num_samples):
    label2id = {"human": 0, "machine": 1}
    with open('data/SubtaskA/subtaskA_train_monolingual.jsonl', 'r') as f:
        train = [json.loads(line) for line in f.readlines()][0:num_samples]
        # train = [{'text': e['text'], 'label': label2id[e['label']]} for e in train]
    with open('data/SubtaskA/subtaskA_dev_monolingual.jsonl', 'r') as f:
        test = [json.loads(line) for line in f.readlines()][0:num_samples]
        # test = [{'text': e['text'], 'label': label2id[e['label']]} for e in test]
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
        labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.long)

        return inputs, labels



def get_prompts_and_labels(sources, targets, tokenizer, max_text_length):
    system_prompt='''Your are an expert to detect whether a given piece of text was likely written by an AI or a human. You should use your understanding of language patterns, nuances, and characteristics typical of AI-generated and human-written texts.
You should also be mindful not to provide any form of judgment or opinion on the content of the text itself, focusing solely on its origin.'''
    question_prompt="Is this text written by a human?"
    prompts = []
    for source in sources:
        conv = get_conv_template("llama-2")
        conv.set_system_message(system_prompt)
        input_ids = tokenizer.encode(source, max_length=max_text_length, add_special_tokens=False, truncation=True)
        truncated_source = tokenizer.decode(input_ids)
        conv.append_message(conv.roles[0], truncated_source+"\n\n"+question_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
    label_to_answer = {0: "No", 1: "Yes"}
    target_texts = []
    for target in targets:
        target_text = label_to_answer[target]
        target_texts.append(target_text)
    return prompts, target_texts


class CausalLMDataset(Dataset):
    def __init__(self, tokenizer, sources, targets, max_length, max_text_length):
        super(CausalLMDataset, self).__init__()
        self.tokenizer = tokenizer
        prompts, targets = get_prompts_and_labels(sources, targets, tokenizer, max_text_length)
        self.sources = prompts
        self.targets = targets
        self.max_length = max_length
        self.has_print = False

    def _tokenize(self, text):
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        full_prompt = self.sources[item] + ' ' + self.targets[item]
        user_prompt = self.sources[item]

        # set a prompt for inputs
        # full_prompt = self.instruction_prompt.format(instruction=self.sources[item]) + self.response_prompt.format(response=self.targets[item])
        # user_prompt = self.response_prompt.format(response=self.targets[item])

        if not self.has_print:
            print(full_prompt, user_prompt)
            self.has_print = True

        tokenized_full_prompt = self._tokenize(full_prompt)
        if (tokenized_full_prompt["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(tokenized_full_prompt["input_ids"]) < self.max_length):
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_full_prompt["attention_mask"].append(1)

        tokenized_user_prompt = self._tokenize(user_prompt)["input_ids"]
        user_prompt_len = len(tokenized_user_prompt)
        labels = [-100 if i < user_prompt_len else token_id for i, token_id in enumerate(tokenized_full_prompt["input_ids"])]

        return torch.tensor(tokenized_full_prompt["input_ids"]), \
            torch.tensor(tokenized_full_prompt["attention_mask"]), \
            torch.tensor(labels)


class CausalLMCollator(object):
    def __init__(self, tokenizer, padding_side='right'):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = tokenizer.eos_token_id
        self.padding_side = padding_side

    def __call__(self, instances):
        input_ids = [e[0] for e in instances]
        attention_masks = [e[1] for e in instances]
        labels = [e[2] for e in instances]

        if self.padding_side == 'left':
            # pad all inputs from left side, this can help batch generation
            reversed_input_ids = [ids.flip(0) for ids in input_ids]
            reversed_attention_masks = [mask.flip(0) for mask in attention_masks]
            reversed_labels = [label.flip(0) for label in labels]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_input_ids = padded_input_ids.flip(1)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(reversed_attention_masks, batch_first=True, padding_value=0)
            padded_attention_masks = padded_attention_masks.flip(1)
            padded_labels = torch.nn.utils.rnn.pad_sequence(reversed_labels, batch_first=True, padding_value=-100)
            padded_labels = padded_labels.flip(1)
        elif self.padding_side == 'right':
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        else:
            raise RuntimeError("Padding side must 'left' or 'right'.")

        return {"input_ids": padded_input_ids, "attention_mask": padded_attention_masks, "labels": padded_labels}

    def _mask(self, lens, max_length):
        mask = torch.arange(max_length).expand(len(lens), max_length) < torch.tensor(lens).unsqueeze(1)
        return mask

