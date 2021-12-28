from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

url = 'https://tiki.vn/tham-tap-yoga-gym-2-lop-tui-dung-tham-tap-yoga-giao-mau-ngau-nhien-p98158374.html?itm_campaign=tiki-reco_UNK_DT_UNK_UNK_tiki-listing_UNK_p-category-mpid-listing-v1_202112270600_MD_PID.98158375&itm_medium=CPC&itm_source=tiki-reco&spid=98158375'


def is_findable_element(driver, attribute, attribute_value, target=None):
    attribute = attribute.upper()

    if target and attribute == 'XPATH':
        raise Exception('XPATH is invalid to find in target')

    driver.implicitly_wait(0)

    finder = target or driver
    result = finder.find_elements(getattr(By, attribute), attribute_value)

    return bool(result)


def crawl(url, amount=9999999):
    """

    Args:
        url: link of product
        amount: amount of comments want to crawl

    Returns:
        list of comments
    """
    result = []

    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    xpath_content_cmt = "//*[contains(@class,'review-comment__content')]"
    xpath_title_cmt = "//*[contains(@class,'review-comment__title')]"
    xpath_next_btn = "//a[@class='btn next']"

    while amount > 0:
        driver.execute_script("window.scrollBy(0,50000)")
        time.sleep(1)
        comments = driver.find_elements_by_xpath(xpath_content_cmt)
        titles = driver.find_elements_by_xpath(xpath_title_cmt)
        for i in range(0, len(comments)):
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
    print(crawl(url, amount=20))
