import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class CohModel(nn.Module):
    def __init__(self,
            max_len,
            bert_dim=768,
            hidden_dropout_prob=0.1,
            num_labels=2
            ):
        super(CohModel, self).__init__()
        self.max_len = max_len
        self.bert_dim = bert_dim
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(bert_dim*3, num_labels)
        
    def forward(self, x, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(
            x.view(-1, self.max_len),
            token_type_ids=token_type_ids.view(-1, self.max_len),
            attention_mask=attention_mask.view(-1, self.max_len)
            )
            
        last_hidden_states = outputs[0]
        # pooled_output = self.dropout(pooled_output)
        mean = torch.mean(last_hidden_states, 1).view(-1, 2, self.bert_dim)
        u = mean[:,0]
        v = mean[:,1]
        concat = torch.cat([u, v, torch.abs(u-v)], -1)

        logits = self.dense(self.dropout(concat))
        
        outputs = (logits, concat,)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs