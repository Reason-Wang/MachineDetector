import torch
from torch import nn
from transformers import AutoConfig, AutoModel


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CustomModel(nn.Module):
    def __init__(self, model_name_or_path, fc_dropout=0.1, target_size=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, target_size)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs

        feature = mean_pooling(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        logits = self.fc(self.fc_dropout(feature))
        return logits

