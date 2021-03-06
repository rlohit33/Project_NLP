# -*- coding: utf-8 -*-
"""Load_Dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QBN4H2s_PBdlcT60amZ_BoEcWrJ5mKQj
"""

import re
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
from datasets import list_datasets, list_metrics, load_dataset, load_metric
import pandas as pd
from google.colab import files

def download_data_tsv(dataframe):
  dataframe.to_csv('threelakh_random_articles.tsv',sep="\t") 
  files.download('threelakh_random_articles.tsv')

def loading_dataset():
    dataset = load_dataset('wikipedia','20200501.en', split='train[:5%]')
    dataframe  = pd.DataFrame(dataset[:3])
    cleaned_dataframe = create_clean_df(dataframe)
    download_data_tsv(cleaned_dataframe)
    return cleaned_dataframe


def convert_non_ascii(text):
  """
  function to covert the non-unicode string literal to unicode
  """
  encoded_string = text.encode("ascii", "ignore")
  decode_string = encoded_string.decode()
  return decode_string


def create_clean_df(dataframe):
  count = 0

  for article_txt in dataframe.text:
    if not article_txt == None :
      article_txt = re.sub(r"&nbsp","",article_txt)
      article_txt = re.sub(r"http\S+","",article_txt)   
      article_txt = re.sub(r"Category:.*","",article_txt)
      article_txt = re.sub(r"References*","",article_txt)
      article_txt = re.sub(r"Extrenal Link*","",article_txt)
      article_txt = re.sub(r"See also*","",article_txt)
      article_txt = re.sub(r"\n"," ",article_txt)  
      article_txt = re.sub(r"\!|\#|\$|\%|\&|\(|\)|\*|\+|\-|\/|\:|\;|\<|\=|\>|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~"," ",article_txt)
      article_txt = re.sub(r" +"," ",article_txt)
      article_txt = article_txt.replace(u'\xa0', u' ')
      if len(article_txt) > 0:
        convert_non_unicode = [convert_non_ascii(article_txt)] # converting non-unicode characters to their unicode equivalent
        article_txt = ' '.join(convert_non_unicode)
    dataframe['text'][count] = article_txt
    count += 1
  return dataframe                     

#df = loading_dataset()

