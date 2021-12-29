import time
import requests
import json
from tqdm import tqdm
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def get_idproduct(link_input):
    index_html = link_input.find(".html?")
    if index_html == -1:
        return '', False
    index_start_id = link_input.rfind("-p",0,index_html)
    if index_start_id == -1:
        return '', False
    id_product = link_input[index_start_id + 2:index_html]
    return id_product, True
    
    

def crawl_from_link(link_input):
    id_product, success = get_idproduct(link_input)
    if not success:
        return [], False
    print(id_product)
    url = "https://tiki.vn/api/v2/reviews?product_id="+str(id_product)
    solditems = requests.get(url, headers={"User-Agent": "curl/7.61.0"})
    data = solditems.json()
    l_review = []
    print(len(data["data"]))
    for item in data["data"]:
        review = item["content"]
        if item["content"] == "" and item["title"] != "":
            review = item["title"]
        l_review.append(review)
    print(len(l_review))
    return l_review, True

def load_url_selenium_tiki(url):
    driver=webdriver.Chrome(executable_path='/usr/bin/chromedriver')
    print("Loading url=", url)
    driver.get(url)
    list_review = []
    product_reviews = driver.find_elements_by_css_selector("[class='style__StyledComment-sc-10ol6xi-5 dOzoKz review-comment']")
    print(product_reviews)
    driver.close()
    return list_review

if __name__ == '__main__':
    list_review = load_url_selenium_tiki("https://tiki.vn/binh-giu-nhiet-bang-thep-khong-gi-lock-lock-vacuum-bottle-lhc6180slv-800ml-p73124602.html?itm_campaign=tiki-reco_UNK_DT_UNK_UNK_tiki-listing_UNK_p-category-mpid-listing-v1_202112270600_MD_PID.73124604&itm_medium=CPC&itm_source=tiki-reco&spid=73124604")
    print(list_review)
        