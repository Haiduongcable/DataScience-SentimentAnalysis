import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, AutoTokenizer
import matplotlib.pyplot as plt
from underthesea import word_tokenize
import regex as re

class PhoBert_Classification(torch.nn.Module):
    def __init__(self, num_class):
        super(PhoBert_Classification, self).__init__()
        self.backbone = AutoModel.from_pretrained("vinai/phobert-base")
        
        self.dense_1 = torch.nn.Linear(in_features = 768, out_features = 128, bias=True)
        self.dense_2 = torch.nn.Linear(in_features = 128, out_features = num_class, bias=True)
        self.dropout1 = nn.Dropout(0.6)
        self.relu =  nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)
        #softmax activation function (Log softmax)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, sent_id):
    
        #get pooler_output of ['CLS'] token from bert output
        cls_hs= self.backbone(sent_id).pooler_output
        x = self.dropout1(cls_hs)
        x = self.dense_1(cls_hs)

        x = self.relu(x)

        x = self.dropout2(x)

        # output layer
        x = self.dense_2(x)

        x = self.softmax(x)

        return x
