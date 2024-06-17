import re
import pandas as pd

def clean(text):
    # lower case
    text = text.lower()
    # clean html entity (&{char};), emoticon (\\{char}), and @user tweet
    text = re.sub(r'&([^;]+);|\\x[a-z0-9]{2}|\\n|@([a-z0-9]+)\s', ' ', text)
    # add white space before after sentence
    text = " "+text+" "
    # clean special char except alphanumeric, whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # clean multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    # clean whitespace at before/after sentence
    text = re.sub(r'^\s+|\s+$', '', text)
    return text

def word_substitute(text, kamus):
    # add white space before after sentence
    text = " "+text+" "
    # iterate kamus
    for word in kamus:
        # replace slay word
        text = text.replace(" "+word[0]+" ", " "+word[1]+" ")
    
    # clean whitespace at before/after sentence
    text = re.sub(r'^\s+|\s+$', '', text)
    print(text)
    return text

def kamus_alay():
    # read kamus alay and convert to array
    df_kamusalay = pd.read_csv(r"D:\Binar\platinum\new_kamusalay (1).csv", header=None, encoding='latin1')
    kamus = [tuple(row) for index, row in df_kamusalay.iterrows()]
    return kamus