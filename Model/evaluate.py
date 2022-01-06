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
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

def evaluate(model, t_dataset_loader):
      
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_groundtruth = []

    # iterate over batches
    for step,batch in enumerate(t_dataset_loader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch


        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)
            
            out_labels = labels.detach().cpu().numpy()
            total_groundtruth.append(out_labels)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0,)
    total_preds = np.argmax(total_preds, axis=1)
    total_preds = np.array(total_preds, dtype = np.int16)
    total_groundtruth = np.concatenate(total_groundtruth, axis = 0)
    total_groundtruth = np.array(total_groundtruth, dtype = np.int16)

    #F1 score
    focus_f1 = f1_score(total_groundtruth, total_preds)
    print("Accuracy: ", accuracy_score(total_groundtruth, total_preds))
    print("F1 score: ", focus_f1)
    print('Recall:', recall_score(total_groundtruth, total_preds))
    print('Precision:', precision_score(total_groundtruth, total_preds))
    print('\n clasification report:\n', classification_report(total_groundtruth,total_preds))
    print('\n confussion matrix:\n',confusion_matrix(total_groundtruth, total_preds))
#     wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
#                         y_true=total_groundtruth, preds=total_preds,
#                         class_names=["Negative", "Positive"])})


    return avg_loss, total_preds, focus_f1


def test_evaluate(model, t_dataset_loader):
      
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_groundtruth = []

    # iterate over batches
    for step,batch in enumerate(t_dataset_loader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch


        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)


            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)
            
            out_labels = labels.detach().cpu().numpy()
            total_groundtruth.append(out_labels)

    # compute the validation loss of the epoch

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0,)
    total_preds = np.argmax(total_preds, axis=1)
    total_preds = np.array(total_preds, dtype = np.int16)
    total_groundtruth = np.concatenate(total_groundtruth, axis = 0)
    total_groundtruth = np.array(total_groundtruth, dtype = np.int16)

    #F1 score
    focus_f1 = f1_score(total_groundtruth, total_preds)
    print("Accuracy: ", accuracy_score(total_groundtruth, total_preds))
    print("F1 score: ", focus_f1)
    print('Recall:', recall_score(total_groundtruth, total_preds))
    print('Precision:', precision_score(total_groundtruth, total_preds))
    print('\n clasification report:\n', classification_report(total_groundtruth,total_preds))
    print('\n confussion matrix:\n',confusion_matrix(total_groundtruth, total_preds))
    wandb.log({"Evaluate conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=total_groundtruth, preds=total_preds,
                        class_names=["Negative", "Positive"])})


    return total_preds, focus_f1

