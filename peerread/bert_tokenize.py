import torch 
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

def token_encode(data_x, data_y, max_length=None, num_sentences=None, sentence_length=None, normalize=False, 
        rel_labels=None, rel_masks=None):

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_shape = None

    if max_length is not None:

        # print(' Original: ',  ' '.join(data_x[0]))
        # print('Tokenized: ', tokenizer.tokenize(' '.join(data_x[0])))
        # print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(data_x[0]))))

        input_ids = []
        token_type_ids = []
        attention_masks = []

        for sample in data_x:
            text = ' '.join(sample)
            encoded_dict = tokenizer(text,
                                    max_length=max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_token_type_ids=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')               
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids)
        token_type_ids = torch.cat(token_type_ids)
        attention_masks = torch.cat(attention_masks)

        tensor_list = (input_ids, token_type_ids, attention_masks)

    if num_sentences is not None and sentence_length is not None:

        # print(' Original: ',  data_x[0][0])
        # print('Tokenized: ', tokenizer.tokenize(data_x[0][0]))
        # print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data_x[0][0])))

        s_input_ids = []
        s_token_type_ids = []
        s_attention_masks = []

        for sample in data_x:
                
            _input_ids = []
            _token_type_ids = []
            _attention_masks = []
            
            for i in range(num_sentences-1):
                if i < len(sample):
                    text_a = sample[i]
                    if i+1 < len(sample):
                        text_b = sample[i+1]
                    else:
                        text_b = ''
                else:
                    text_a = ''
                    text_b = ''
                encoded_dict = tokenizer.encode_plus(text_a, text_b,
                                                    max_length=sentence_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt') 
                _input_ids.append(encoded_dict['input_ids'])
                _token_type_ids.append(encoded_dict['token_type_ids'])
                _attention_masks.append(encoded_dict['attention_mask'])
                
            _input_ids = torch.cat(_input_ids)
            _token_type_ids = torch.cat(_token_type_ids)
            _attention_masks = torch.cat(_attention_masks)
            
            s_input_ids.append(_input_ids)
            s_token_type_ids.append(_token_type_ids)
            s_attention_masks.append(_attention_masks)
            
        s_input_ids = torch.stack(s_input_ids)
        s_token_type_ids = torch.stack(s_token_type_ids)
        s_attention_masks = torch.stack(s_attention_masks)

        input_shape = tuple(s_input_ids.shape)[1:]

        tensor_list = (s_input_ids, s_token_type_ids, s_attention_masks)

    if max_length is not None and num_sentences is not None and sentence_length is not None:
        tensor_list = (input_ids, token_type_ids, attention_masks, s_input_ids, s_token_type_ids, s_attention_masks)

    labels = torch.Tensor(data_y)

    if normalize:
        low, high = 1, 5
        labels = (labels - low) / (high - low)

    tensor_list = tensor_list + (labels,)
    
    if rel_labels is not None:
        rel_labels = rel_labels.type(torch.long)
        tensor_list = tensor_list +(rel_labels,)
    if rel_masks is not None:
        rel_masks = rel_masks.type(torch.bool)
        tensor_list = tensor_list +(rel_masks,)

    dataset = TensorDataset(*tensor_list)
    
    return dataset, input_shape