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
from model.bert_classification import PhoBert_Classification
from utils import load_config
from preprocess import clean_review

class Sentiment_Analysis():
    def __init__(self, num_class):
        self.config = load_config()
        self.model = PhoBert_Classification(num_class)
        self.path_weights = "weights/Best_weights_f1.pt"
        if self.config["model"]["device"] == "cuda":
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.path_weights,map_location=self.device))
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.path_weights,map_location=self.device))
        if self.config["model"]["inference"]:
            self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.max_len = 200
        
    def inference(self, review_string):
        #cleam
        clean_string, success = clean_review(review_string)
        if not success:
            return False
        #encoding
        inference_seq = torch.tensor([self.tokenizer.encode(clean_string)])
        sent_id = inference_seq.to(self.device)
        preds = self.model(sent_id)
        preds = preds.detach().cpu().numpy()
        
        class_predict = np.argmax(preds, axis = 1)[0]
        return class_predict
        
if __name__ == '__main__':
    sentiment_analysis = Sentiment_Analysis(2)
    string = "đẹp, giá cả phải chăng, chất lượng lắm"
    sentiment_analysis.inference(string)