import pandas as pd
import re
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English


def sentence_token(text):
  sentences=[]
  nlp = English()
  nlp.add_pipe(nlp.create_pipe('sentencizer'))
  doc = nlp(text)
  sentences = [sent.string.strip() for sent in doc.sents]
  return sentences

def pre_train_textfile(dataframe):
  for i in range(len(dataframe)):
    lst = sentence_token(dataframe['text'][i])
    for line in lst:
      # writing to file
      file1 = open('pretrain.txt', 'a')
      file1.writelines(line+"\n")
    file1.writelines("\n")
    file1.close()
  
#pre_train_textfile(df)



