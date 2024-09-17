import sys
import pandas as pd
import numpy as np
import requests
import bs4
from bs4 import BeautifulSoup
import torch.nn as nn
import torch
from pytorch_pretrained_bert import BertTokenizer #, BertModel
from transformers import BertModel
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
import Fact_Check

from flask import Flask, request, jsonify
app = Flask(__name__)
# Model
class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        # _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        outputs = self.bert(tokens, attention_mask=masks)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba

# Preprocessing 

# Imporing tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def Punctuation(string):

    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")

    # return string without punctuation
    return string

def get_text(url):
    try:
        result=requests.get(str(url))
    except Exception:
        print("error in scraping url")
        return None
    src=result.content
    soup=BeautifulSoup(src,'lxml')
    text=[]
    for p_tag in soup.find_all('p'):
        text.append(p_tag.text)
    text = Punctuation(str(text))
    return text


# loading model
# cange path as per your requirement
path='nb_state256.pth'
model = BertBinaryClassifier()
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
model.load_state_dict(torch.load(path))
model.eval()

def test(article,model):
    bert_predicted = []
    all_logits = []
    test_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:255], [article]))
    test_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
    test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=256, truncating="post", padding="post", dtype="int")
    test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]
    test_masks_tensor = torch.tensor(test_masks)
    test_tokens_ids = torch.tensor(test_tokens_ids)
    with torch.no_grad():
        logits = model(test_tokens_ids, test_masks_tensor)
        numpy_logits = logits.cpu().detach().numpy()
        if(numpy_logits[0,0] > 0.5):
            return 'Fake'
        else:
            return 'True'


# def answer(url,model):
#     article = get_text(url)
#     ans = test(article,model)
#     Fact_Check.fact_check_article(url)
#     return ans


@app.route('/answer', methods=['POST'])
def answer():
    data = request.get_json()
    url = data.get('url')  # Sử dụng model có sẵn trong code

    if not url:
        return jsonify({"error": "URL and model are required"}), 400

    article = get_text(url)
    ans = test(article, model)

    # Gọi hàm fact_check_article với URL và lấy kết quả
    fact_check_result = Fact_Check.fact_check_article(url)

    # Trả về cả ans và fact_check_result trong response
    return jsonify({
        "answer": ans,
        "fact_check": fact_check_result
    })


if __name__ == '__main__':
    app.run(debug=True)


# url = 'https://www.statesman.com/story/news/politics/politifact/2024/08/04/politifact-claim-that-trump-would-cut-social-security-lacks-basis/74647925007/'
# print(answer(url,model))


