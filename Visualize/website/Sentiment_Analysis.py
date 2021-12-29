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
from .model.bert_classification import PhoBert_Classification
from .utils import load_config, check_validation_url
from .preprocess import clean_review
from .crawl import crawl
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

class Sentiment_Analysis():
    def __init__(self, num_class):
        self.config = load_config()
        self.model = PhoBert_Classification(num_class)
        self.path_weights = "website/weights/Best_weights_f1.pt"
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
        self.classes = ["Negative", "Positive"]
        options = Options()
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        
        
    def predict_1_sentence(self, clean_string):
        inference_seq = torch.tensor([self.tokenizer.encode(clean_string)])
        sent_id = inference_seq.to(self.device)
        preds = self.model(sent_id)
        preds = preds.detach().cpu().numpy()
        class_predict = np.argmax(preds, axis = 1)[0]
        return class_predict
        
    def inference_link(self, link_product):
        print("Heyyy")
        l_review = crawl(link_product,self.driver, amount=100)
        print("Crawl done")
        if len(l_review) == 0:
            return [], []
        l_used_review = []
        l_predict = []
        for index, review in enumerate(l_review):
            clean_string, success = clean_review(review)
            if not success:
                continue
            else:
                class_predict = self.predict_1_sentence(clean_string)
                l_predict.append(class_predict)
                l_used_review.append(review)
        return l_predict, l_used_review
                
    def inference(self, review_string):
        if check_validation_url(review_string):
            return "link", self.inference_link(review_string)
        else:
            return "text", self.inference_review(review_string)
        
        
    def inference_review(self, review_string):
        tmp_log = review_string.split(" ")
        if len(tmp_log) == 1 and len(tmp_log[0]) >= 7:
            return '', False
        #cleam
        clean_string, success = clean_review(review_string)
        if not success:
            return '', False
        #encoding python main.py
        inference_seq = torch.tensor([self.tokenizer.encode(clean_string)])
        sent_id = inference_seq.to(self.device)
        preds = self.model(sent_id)
        preds = preds.detach().cpu().numpy()
        prob_class_0 = round(float(preds[0][0]),2)
        prob_class_1 = round(float(preds[0][1]),2)
        class_predict = np.argmax(preds, axis = 1)[0]
        return class_predict, True
        
if __name__ == '__main__':
    sentiment_analysis = Sentiment_Analysis(2)
    string = "https://tiki.vn/binh-giu-nhiet-lock-lock-lhc1439-dung-tich-1000ml-p16341786.html?spid=16341788"
    type_str, result = sentiment_analysis.inference(string)