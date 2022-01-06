import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from underthesea import word_tokenize
import regex as re

device = torch.device("cuda")
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from evaluate import test_evaluate, evaluate
from model.bert_classification import PhoBert_Classification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def train(model):
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds=[]
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch
        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds
if __name__ == '__main__':
    path_dataset = "/kaggle/input/datasetv1mergev2/Dataset_24_12_version1Mergeversion2.xlsx"
    dataframe = pd.read_excel(path_dataset, sheet_name = 'Dataset')
    train_text, tmp_text, train_labels, tmp_labels = train_test_split(dataframe['Review'], dataframe['Label'], 
                                                                    random_state=2021, 
                                                                    test_size=0.2, 
                                                                    stratify=dataframe['Label'])


    val_text, test_text, val_labels, test_labels = train_test_split(tmp_text, tmp_labels, 
                                                                        random_state=2021, 
                                                                        test_size=0.5, 
                                                                        stratify=tmp_labels)

    train_text = train_text.astype(str)
    val_text = val_text.astype(str)
    test_text = test_text.astype(str)
    
    
    model = PhoBert_Classification(2)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    # tokenize and encode sequences in the training set
    MAX_LENGTH = 200
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length = MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length = MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )

    # # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    batch_size = 32

    train_data = TensorDataset(train_seq, train_mask, train_y)

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_seq, val_mask, val_y)

    val_sampler = SequentialSampler(val_data)

    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = SequentialSampler(test_data)

    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)
    
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr = 1e-5) 
    class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
    weights= torch.tensor(class_weights,dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy  = nn.NLLLoss(weight=weights) 

    # set initial loss to infinite
    best_valid_loss = float('inf')
    best_valid_f1 = 0

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]
    n_epochs = []
    d = 0

    warmup_nepochs = 10
    finetune_nepochs = 150
    for param in model.backbone.parameters():
        param.requires_grad = False
            
    #for each epoch
    for epoch in range(warmup_nepochs):
        print("Start")
        print('\n Warmup Epoch {:} / {:}'.format(epoch + 1, warmup_nepochs))
        
        #train model
        train_loss, _ = train(model)
        print({"Loss train": train_loss})
        
        #evaluate model
        valid_loss, _, f1_value = evaluate(model, val_dataloader)
        print({"Loss val": valid_loss})
        
        print({"F1 score": f1_value})
        
        #save the best model
        if f1_value > best_valid_f1:
            best_valid_f1 = f1_value
            torch.save(model.state_dict(), '/kaggle/working/Best_weights_f1.pt')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        torch.save(model.state_dict(), '/kaggle/working/Lass_weights_f1.pt')
        
        # append training and validation loss
        d+=1
        n_epochs.append(d)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        

    for param in model.backbone.parameters():
        param.requires_grad = True
            
    for epoch in range(finetune_nepochs):
        print("Start")
        print('\n FineTune Epoch {:} / {:}'.format(epoch + 1, finetune_nepochs))

        #train model
        train_loss, _ = train(model)
        print({"Loss train": train_loss})
        #evaluate model
        valid_loss, _, f1_value = evaluate(model, val_dataloader)
        print({"Loss val": valid_loss})
        print({"F1 score": f1_value})
        #save the best model
        if f1_value > best_valid_f1:
            best_valid_f1 = f1_value
            torch.save(model.state_dict(), '/kaggle/working/Best_weights_f1.pt')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        torch.save(model.state_dict(), '/kaggle/working/Lass_weights_f1.pt')
        
        # append training and validation loss
        d+=1
        n_epochs.append(d)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

