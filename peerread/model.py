import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig

from discourse.model import CohModel

class BertRegression(nn.Module):
    def __init__(self, bert_dim=768, hidden_dropout_prob=0.1):
        super(BertRegression, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(bert_dim, 1)

    def forward(self, x, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)

        outputs = (logits, pooled_output)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

class BertCNN(nn.Module):
    def __init__(self, bert_dim=768, cnn_filters=128, kernel_size=5, hidden_dim = 100):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.conv1 = nn.Conv1d(bert_dim, cnn_filters, kernel_size)
        self.fc = nn.Linear(cnn_filters, 1)

    def forward(self, x, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
            )

        last_hidden_state = outputs[0]
        cnn = self.conv1(last_hidden_state.permute(0, 2, 1))
        gmp, _ = torch.max(cnn, -1)

        logits = self.fc(gmp)

        outputs = (logits, gmp)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

class BertRNN(nn.Module):
    def __init__(self, bert_dim=768, hidden_dim=100):
        super(BertRNN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.rnn = nn.GRU(bert_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
            )

        last_hidden_state = outputs[0]
        outputs, _ = self.rnn(last_hidden_state.permute(1, 0, 2))
        last = outputs[-1]

        logits = self.fc(last)

        outputs = (logits, last)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

class BertDAN(nn.Module):
    def __init__(self, bert_dim=768):
        super(BertDAN, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(bert_dim, 1)

    def forward(self, x, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(
            x,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
            )

        last_hidden_state = outputs[0]
        last = torch.mean(last_hidden_state, 1)

        logits = self.fc(last)

        outputs = (logits, last)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

class CombinedModel(nn.Module):
    def __init__(self,
                 input_shape,
                 bert_dim=768,
                 hidden_dropout_prob=0.1,
                 main_model_checkpoint=None,
                 coh_checkpoint=None,
                ):
        super(CombinedModel, self).__init__()
        self.input_shape = input_shape
        self.bert_dim = bert_dim
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(bert_dim*2, 1)

        if coh_checkpoint is not None:
            print("Loading coherence model ...")
            self.coh_model = CohModel()
            checkpoint = torch.load(coh_checkpoint)
            print('Epoch {}'.format(checkpoint['epoch']))
            print('Validation Accuracy: {0:.2f}'.format(checkpoint['best_val_acc']))
            print('Test Accuracy: {0:.2f}'.format(checkpoint['test_acc']))
            self.coh_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.coh_model = CohModel()

        if main_model_checkpoint is not None:
            print("Loading main model ...")
            self.main_model = BertRegression()
            checkpoint = torch.load(main_model_checkpoint)
            print('Epoch  {}'.format(checkpoint['epoch']))
            print('Validation Accuracy: {0:.2f}'.format(checkpoint['best_val_loss']))
            print('Test Accuracy: {0:.2f}'.format(checkpoint['test_loss']))
            self.main_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.main_model = BertRegression()

    def forward(self, x, token_type_ids, attention_mask, 
                s_x, s_token_type_ids, s_attention_mask, 
                labels=None,):
    
        outputs = self.main_model(x,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        doc_infor = outputs[1]

        s_outputs = self.coh_model(s_x.view(-1, self.input_shape[-1]),
                                   token_type_ids=s_token_type_ids.view(-1, self.input_shape[-1]),
                                   attention_mask=s_attention_mask.view(-1, self.input_shape[-1]))

        coh_infor = s_outputs[1].view(-1, self.input_shape[0], self.bert_dim)
        coh_infor, _ = torch.max(coh_infor, 1)

        concat = torch.cat((doc_infor, coh_infor), 1)
        concat = self.dropout(concat)

        logits = self.dense(concat)

        outputs = (logits, concat)        

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs


