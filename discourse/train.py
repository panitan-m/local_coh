import torch
import random
import os, time, datetime
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split 
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from model import CohModel

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = pd.read_csv('./sentence_pair_rel.csv')
no_rel_df = df[df.relation=='none'].copy()
rel_df = df[df.relation!='none'].copy()

label_map = {}
for label in df.relation.unique():
    if label == 'none':
        label_map[label] = 0
    else:
        label_map[label] = 1

no_rel_df.loc[:, 'relation'] = no_rel_df.relation.map(label_map)
rel_df.loc[:, 'relation'] = rel_df.relation.map(label_map)

random_state = 42

no_rel_train_df, no_rel_test_df = train_test_split(no_rel_df, test_size=1/4, random_state=random_state)
no_rel_train_df, no_rel_dev_df = train_test_split(no_rel_train_df, test_size=1/10, random_state=random_state)
rel_train_df, rel_test_df = train_test_split(rel_df, test_size=1/4, random_state=random_state)
rel_train_df, rel_dev_df = train_test_split(rel_train_df, test_size=1/10, random_state=random_state)


print(len(no_rel_train_df), len(no_rel_dev_df), len(no_rel_test_df))
print(len(rel_train_df), len(rel_dev_df), len(rel_test_df))

train_df = no_rel_train_df.append(rel_train_df).sample(frac=1, random_state=random_state)
dev_df = no_rel_dev_df.append(rel_dev_df).sample(frac=1, random_state=random_state)
test_df = no_rel_test_df.append(rel_test_df).sample(frac=1, random_state=random_state)

print(len(train_df), len(dev_df), len(test_df))
print(train_df.iloc[0].text_a)
print(dev_df.iloc[0].text_a)
print(test_df.iloc[0].text_a)

text_a = {}
text_b = {}
labels = {}

dataset_types = ['train', 'val', 'test']

for df_type, df in zip(dataset_types, [train_df, dev_df, test_df]):
    text_a[df_type] = df.text_a.values
    text_b[df_type] = df.text_b.values
    labels[df_type] = df.relation.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = {}
token_type_ids = {}
attention_masks = {}

max_len = 165

for dataset_type in dataset_types:
    _input_ids = []
    _token_type_ids = []
    _attention_masks = []
    for a, b in zip(text_a[dataset_type], text_b[dataset_type]):
        __input_ids = []
        __token_type_ids = []
        __attention_masks = []
        for text in [a, b]:
            encoded_dict = tokenizer.encode_plus(text,
                                                add_special_tokens=True,
                                                max_length=max_len,
                                                truncation=True,
                                                pad_to_max_length=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                return_tensors='pt')
            __input_ids.append(encoded_dict['input_ids'])
            __token_type_ids.append(encoded_dict['token_type_ids'])
            __attention_masks.append(encoded_dict['attention_mask'])

        __input_ids = torch.cat(__input_ids)
        __token_type_ids = torch.cat(__token_type_ids)
        __attention_masks = torch.cat(__attention_masks)

        _input_ids.append(__input_ids)
        _token_type_ids.append(__token_type_ids)
        _attention_masks.append(__attention_masks)

    input_ids[dataset_type] = torch.stack(_input_ids)
    token_type_ids[dataset_type] = torch.stack(_token_type_ids)
    attention_masks[dataset_type] = torch.stack(_attention_masks)
    labels[dataset_type] = torch.tensor(labels[dataset_type])

train_dataset = TensorDataset(input_ids['train'], token_type_ids['train'], attention_masks['train'], labels['train'])
val_dataset = TensorDataset(input_ids['val'], token_type_ids['val'], attention_masks['val'], labels['val'])
test_dataset = TensorDataset(input_ids['test'], token_type_ids['test'], attention_masks['test'], labels['test'])

# with open('essay_1.TensorDataset', 'rb') as f:
#     peerread_dataset = pickle.load(f)

# train_dataset = train_dataset.__add__(peerread_dataset)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))
print('{:>5,} test samples'.format(len(test_dataset)))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

best_val_acc = 0
result_dir = './results'
if not os.path.exists(result_dir): os.makedirs(result_dir)

for batch_size in [8, 16, 32]:
    for lr in [1e-5, 2e-5, 3e-5]:

        batch_size = batch_size
        train_dataloader = DataLoader(train_dataset,
                                    sampler=RandomSampler(train_dataset),
                                    batch_size=batch_size)
        validation_dataloader = DataLoader(val_dataset,
                                        sampler=SequentialSampler(val_dataset),
                                        batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset,
                                        sampler=SequentialSampler(test_dataset),
                                        batch_size=batch_size)
        
        model = CohModel(max_len=max_len)
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

        for epoch_i in range(0, epochs):
            
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            
            t0 = time.time()

            total_train_accuracy = 0
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
                loss, logits, _ = model(b_input_ids, 
                                    token_type_ids=b_input_type_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
                total_train_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_train_accuracy += flat_accuracy(logits, label_ids)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            print("")
            avg_train_accuracy = total_train_accuracy / len(train_dataloader)
            print("  Traning Accuracy: {0:.2f}".format(avg_train_accuracy))
                
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)
            
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            
            print("")
            print("Running Validation...")
            
            t0 = time.time()
            
            model.eval()
            
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            
            for batch in validation_dataloader:
                
                b_input_ids = batch[0].to(device)
                b_input_type_ids = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
                
                with torch.no_grad():
                    (loss, logits, _) = model(b_input_ids,
                                        token_type_ids=b_input_type_ids,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
            
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = format_time(time.time() - t0)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            model.eval()
            
            total_test_accuracy = 0
            total_test_loss = 0
            nb_test_steps = 0
            
            for batch in test_dataloader:
                
                b_input_ids = batch[0].to(device)
                b_input_type_ids = batch[1].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)
                
                with torch.no_grad():
                    (loss, logits, _) = model(b_input_ids,
                                        token_type_ids=b_input_type_ids,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                
                total_test_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_test_accuracy += flat_accuracy(logits, label_ids)
                
            avg_test_accuracy = total_test_accuracy / len(test_dataloader)
            print("  Test Accuracy: {0:.2f}".format(avg_test_accuracy))
            
            avg_test_loss = total_test_loss / len(test_dataloader)
            test_time = format_time(time.time() - t0)
            print("  Test Loss: {0:.2f}".format(avg_test_loss))
            print("  Test took: {:}".format(test_time))
            
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Accur.': avg_train_accuracy,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Test Accur.': avg_test_accuracy,
                    'Test Loss': avg_test_loss,
                    'Training Time': training_time,
                    'Validation Time': validation_time,
                    'Test Time': test_time,
                }
            )

            if avg_val_accuracy >= best_val_acc:
                print("Validion accuracy increased ({:.2f} --> {:.2f}). Saving model ...".format(best_val_acc, avg_val_accuracy))
                best_val_acc = avg_val_accuracy
                checkpoint = {
                    'epoch': epoch_i + 1,
                    'best_val_acc': best_val_acc,
                    'test_acc': avg_test_accuracy,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(result_dir, 'best_model.pt'))
                

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        pd.set_option('precision', 2)
        df_stats = pd.DataFrame(data=training_stats)
        df_stats = df_stats.set_index('epoch')

        file_name = 'b{}_lr{}.tsv'.format(batch_size, lr)
        df_stats.to_csv(os.path.join(result_dir, file_name), sep='\t')

