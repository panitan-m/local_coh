import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class CohModel(nn.Module):
    def __init__(self,
            bert_dim=768,
            hidden_dropout_prob=0.1,
            num_labels=2
            ):
        super(CohModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(bert_dim, num_labels)
        
    def forward(self, x, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
            )
            
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        
        outputs = (logits, pooled_output,)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs