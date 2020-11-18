import os, sys
import random
import time
import pickle
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, BertTokenizer
from transformers import get_linear_schedule_with_warmup

from utils import format_time
from bert_tokenize import token_encode
from model import BertRegression, BertCNN, BertRNN, BertDAN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='reg', choices=['reg', 'cnn', 'rnn', 'dnn'], help="Model type (default=reg)")
parser.add_argument("--training_ratio", type=float, default=1.0, help="Training ratio (default=1.0)")
args = parser.parse_args()

def models(model_text):
    if model_text == 'cnn': return BertCNN()
    elif model_text == 'rnn': return BertRNN()
    elif model_text == 'dan': return BertDAN()
    elif model_text == 'reg': return BertRegression()
    else: return None,None

with open('./introduction_data', 'rb') as f:
    data_x, data_y = pickle.load(f)

data_x_paragraph = []
data_y_paragraph = []

for paper, score in zip(data_x, data_y):
    for paragraph in paper:
        if len(paragraph.sentences) > 1:
            data_x_paragraph.append([' '.join([word.text for word in sent.words]) for sent in paragraph.sentences])
            data_y_paragraph.append(score)

print('Total dataset:', len(data_x_paragraph), len(data_y_paragraph))

if torch.cuda.is_available():

    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    sys.exit(-1)

max_length = 512

dataset, input_shape = token_encode(data_x_paragraph, data_y_paragraph, max_length=max_length, normalize=True)

# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# fold = 0

for training_seed in range(42, 52):

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=training_seed)
    # train_dataset = TensorDataset(*dataset[train_idx])
    # val_dataset = TensorDataset(*dataset[val_idx])

    if args.training_ratio < 1.0:
        train_dataset, _ = train_test_split(train_dataset, train_size=args.training_ratio, random_state=training_seed)

    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(val_dataset)))
    print('{:>5,} test samples'.format(len(test_dataset)))

    def gen_hp():
        for batch_size in [8]:
            for lr in [3e-5]:
                yield batch_size, lr

    best_val_loss = np.Inf
    result_dir = './results/bert_{}/{}'.format(args.model, args.training_ratio)
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    for batch_size, lr in gen_hp():

        train_dataloader = DataLoader(train_dataset,
                                    sampler=RandomSampler(train_dataset),
                                    batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset,
                                        sampler=SequentialSampler(val_dataset),
                                        batch_size=batch_size)                
        test_dataloader = DataLoader(test_dataset,
                                        sampler=SequentialSampler(test_dataset),
                                        batch_size=batch_size)

        model = models(args.model)
        model.cuda()

        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6)

        epochs = 10
        total_steps = len(train_dataloader) * epochs
        num_warmup_steps = int(0.06 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps, 
                                                    num_training_steps=total_steps)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []

        total_t0 = time.time()

        # weight = weight.to(device)

        for epoch_i in range(0, epochs):
            
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            
            t0 = time.time()
            
            total_train_loss = 0
            model.train()
            
            for step, batch in enumerate(train_dataloader):
                
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                    
                b_input_ids = batch[0].to(device)
                b_input_type_ids = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
                
                model.zero_grad()
                (loss, _, _) = model(b_input_ids, 
                                    token_type_ids=b_input_type_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)
            
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            
            print("")
            print("Running Validation...")
            
            t0 = time.time()
            
            model.eval()
            
            total_eval_loss = 0
            
            for batch in validation_dataloader:
                
                b_input_ids = batch[0].to(device)
                b_input_type_ids = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
                
                with torch.no_grad():
                    (loss, _, _) = model(b_input_ids,
                                        token_type_ids=b_input_type_ids,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                
                total_eval_loss += loss.item()
            
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            model.eval()
            
            total_test_loss = 0
            
            for batch in test_dataloader:
                
                b_input_ids = batch[0].to(device)
                b_input_type_ids = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
                
                with torch.no_grad():
                    (loss, _, _) = model(b_input_ids,
                                        token_type_ids=b_input_type_ids,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                
                total_test_loss += loss.item()
            
            avg_test_loss = total_test_loss / len(test_dataloader)
            test_time = format_time(time.time() - t0)
            print("  Test Loss: {0:.2f}".format(avg_test_loss))
            print("  Test took: {:}".format(test_time))
            
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Test Loss': avg_test_loss,
                    'Training Time': training_time,
                    'Validation Time': validation_time,
                    'Test Time': test_time
                }
            )

            if avg_val_loss <= best_val_loss:
                print("Validion loss decreased ({:.2f} --> {:.2f}). Saving model ...".format(best_val_loss, avg_val_loss))
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch_i + 1,
                    'best_val_loss': best_val_loss,
                    'test_loss': avg_test_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(result_dir, 'model_{}.pt'.format(training_seed)))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        df_stats = pd.DataFrame(data=training_stats)
        df_stats = df_stats.set_index('epoch')

        file_name = 'seed_{}.tsv'.format(training_seed)
        df_stats.to_csv(os.path.join(result_dir, file_name), sep='\t')