from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

def is_findable_element(driver, attribute, attribute_value, target=None):
    attribute = attribute.upper()

    if target and attribute == 'XPATH':
        raise Exception('XPATH is invalid to find in target')

    driver.implicitly_wait(0)

    finder = target or driver
    result = finder.find_elements(getattr(By, attribute), attribute_value)

    return bool(result)

def get_num_review(element_num_review):
    tmp = element_num_review.split(" ")
    num_review = int(tmp[1])
    return num_review

def crawl(url,driver, amount=9999999):
    """

    Args:
        url: link of product
        amount: amount of comments want to crawl

    Returns:
        list of comments
    """
    result = []

    
    driver.get(url)

    xpath_content_cmt = "//*[contains(@class,'review-comment__content')]"
    xpath_title_cmt = "//*[contains(@class,'review-comment__title')]"
    xpath_num_review = "//*[contains(@class,'number')]"
    xpath_next_btn = "//a[@class='btn next']"
    # element_num_review = driver.find_elements_by_xpath(xpath_num_review)[0].text
    # num_review = get_num_review(element_num_review)
    # if amount  num_review:
        
    while amount > 0:
        driver.execute_script("window.scrollBy(0,50000)")
        time.sleep(1)
        comments = driver.find_elements_by_xpath(xpath_content_cmt)
        titles = driver.find_elements_by_xpath(xpath_title_cmt)
        for i in range(0, len(comments)):
            print(amount)
            amount -= 1
            if len(comments[i].text) > 0:
                result.append(comments[i].text)
            else:
                result.append(titles[i].text)
        if not is_findable_element(driver, 'xpath', xpath_next_btn):
            break

        next_btn = driver.find_element_by_xpath(xpath_next_btn)
        driver.execute_script("arguments[0].click();", next_btn)

    return result


if __name__ == '__main__':
    url = "https://tiki.vn/binh-giu-nhiet-lock-lock-lhc1439-dung-tich-1000ml-p16341786.html?spid=16341788"
    time_s = time.time()
    result = crawl(url, amount = 100)
    print("Crawl done in : ", time.time() - time_s)
    print(len(result))
    print(result)
