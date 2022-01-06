import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import json
from tqdm import tqdm
import pandas as pd
from csv import writer


if __name__ == '__main__':

    # s = Service("chromedriver.exe")
    # driver = webdriver.Chrome(service=s)
    # driver = webdriver.Chrome()
    driver = webdriver.Chrome(executable_path="/usr/lib/chromium-browser/chromedriver")
    dict_product_category = {
        "beauty": "https://tiki.vn/lam-dep-suc-khoe/c1520",
        "SmartPhone": "https://tiki.vn/dien-thoai-may-tinh-bang/c1789"
    }

    RAN_NUM_PAGE = [1, 2] # điền số trang

    product_ids = []

    for product_category in dict_product_category:
        for num in range(RAN_NUM_PAGE[0], RAN_NUM_PAGE[1] + 1):
            driver.get(dict_product_category[product_category] + "?page=" + str(num))
            time.sleep(3)
            all_product_per_page = driver.find_elements(By.CLASS_NAME, "product-item")
            for product in all_product_per_page:
                product_ids.append(json.loads(product.get_attribute("data-view-content"))["click_data"]["id"])

    count = pd.read_csv('Comment_Tiki_Data.csv').shape[0]
    recods = 0

    with open('Comment_Tiki_Data.csv', 'a+', newline='',encoding='utf-8') as f:
        csv_writer = writer(f)
        for ids in tqdm(product_ids):
            url = "https://tiki.vn/api/v2/reviews?product_id="+str(ids)
            solditems = requests.get(url, headers={"User-Agent": "curl/7.61.0"})
            data = solditems.json()
            for i in data["data"]:
                count+=1
                recods+=1
                csv_writer.writerow([count,i["content"]])

    print("adding "+str(recods) + " records")
    
    

    driver.quit()

# https://tiki.vn/api/v2/reviews?product_id=126218820
