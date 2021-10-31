# Databricks notebook source
!pip install transformers

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### 使用できるデバイスの数を確認 

# COMMAND ----------

from tensorflow.python.client import device_lib
import multiprocessing
print(multiprocessing.cpu_count())
device_lib.list_local_devices()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### デバイスIDをここに追加します

# COMMAND ----------

device_ids = [0,1,2,3]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### データの取得

# COMMAND ----------

!wget https://dem-primary-tweets.s3.amazonaws.com/PeteForAmerica.1574004110.txt

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のコピー先のアドレス部分は適宜変更してください。

# COMMAND ----------

# MAGIC %fs
# MAGIC 
# MAGIC cp file:/databricks/driver/PeteForAmerica.1574004110.txt dbfs:/Users/takaaki.yayoi@databricks.com/Pete2.txt

# COMMAND ----------

import pandas as pd
df = pd.read_json('/dbfs/Users/takaaki.yayoi@databricks.com/Pete2.txt', lines=True)
df

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### PytorchのHuggingfaceトランスフォーマーを使用

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_batch = tokenizer(
["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt")

pt_outputs = pt_model(**pt_batch)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=1)
pt_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 単一のデータファイルに対して複数GPUを用いて推論

# COMMAND ----------

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import csv
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import urllib
import json
import glob
import os
from torch.utils.data import Dataset

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# 現状、同じファイルを複数回指定しています
def get_all_files():
  file_list = ['/dbfs/Users/takaaki.yayoi@databricks.com/Pete2.txt',
               '/dbfs/Users/takaaki.yayoi@databricks.com/Pete2.txt',
               '/dbfs/Users/takaaki.yayoi@databricks.com/Pete2.txt']
  return(file_list)


class TextLoader(Dataset):
    def __init__(self, file=None, transform=None, target_transform=None, tokenizer=None):
        print(file)
        self.file = pd.read_json(file, lines=True)
        self.file = self.file
        self.file = tokenizer(list(self.file['full_text']), padding=True, truncation=True, max_length=512, return_tensors='pt')
        self.file = self.file['input_ids']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        data = self.file[idx]
        return(data)

      
class SentimentModel(nn.Module):
    # 使用するモデル

    def __init__(self):
        super(SentimentModel, self).__init__()
        #print("------------------- Initializing once ------------------")
        self.fc = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def forward(self, input):
        #print(input)
        output = self.fc(input)
        pt_predictions = nn.functional.softmax(output.logits, dim=1)
        #print("\tIn Model: input size", input.size())
        return(pt_predictions)
      

dev = 'cuda'
if dev == 'cpu':
  device = torch.device('cpu')
  device_staging = 'cpu:0'
else:
  device = torch.device('cuda')
  device_staging = 'cuda:0'
  
tokenizer = AutoTokenizer.from_pretrained(MODEL)

all_files = get_all_files()
model3 = SentimentModel()
try:
      # device_idsパラメーターを除外した場合、利用できる全てのデバイス(GPU)を選択します
      model3 = nn.DataParallel(model3, device_ids=device_ids) 
      model3.to(device_staging)
except:
      torch.set_printoptions(threshold=10000)
      
for file in all_files:
    data = TextLoader(file=file, tokenizer=tokenizer)
    train_dataloader = DataLoader(data, batch_size=120, shuffle=False) # ShuffleはFalseに設定する必要があります
    out = torch.empty(0,0)
    for data in train_dataloader:
        input = data.to(device_staging)
        #print(len(input))
        if(len(out) == 0):
          out = model3(input)
        else:
          output = model3(input)
          with torch.no_grad():
            out = torch.cat((out, output), 0)
            
    df = pd.read_json(file, lines=True)['full_text']
    res = out.cpu().numpy()
    df_res = pd.DataFrame({ "text": df, "negative": res[:,0], "positive": res[:,1]})
    print(df_res)

# COMMAND ----------

# MAGIC %md
# MAGIC # END
