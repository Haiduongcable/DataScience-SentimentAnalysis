import pandas as pd 
import numpy as np 
import os 
import time 
import regex as re
import math
from underthesea import word_tokenize
from utils import remove_html, remove_emojis, covert_unicode, lowercase_remove_noise_character


def clean_review(review_str):
    clean_string = review_str.replace("\n","")
    clean_string = " ".join(clean_string.split())
    clean_string = remove_html(clean_string)
    clean_string = remove_emojis(clean_string)
    unicode_string = covert_unicode(clean_string)
    sign_string = lowercase_remove_noise_character(unicode_string)
    sign_string = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]','',sign_string)
    sign_string = " ".join(sign_string.split())
    if len(sign_string) <= 1:
        return '', False
    token_string = word_tokenize(sign_string, format="text")
    return token_string, True
        
