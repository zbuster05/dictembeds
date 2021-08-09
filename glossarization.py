# type: ignore
# pylint: disable=no-member

from execute import Engine
from nltk.tokenize import word_tokenize
from collections import defaultdict

import math
import re

# model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a2fc9:B_VAL::0:24900:0.8616854640095484"
# model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a5029:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-cf8cd:ROUTINE::0:20000"
model_path = "./training/bart_enwiki-kw_summary-3dee1:B_VAL::0:47200:1.8260153889656068"

TFIDF_FINAL_INCLUDE = 50 # "important" words to include

with open("./textbook.txt", "r") as data:
    data_text = data.read()

documents = [i.strip() for i in list(filter(lambda x:(x!='' and len(x)>1000), re.sub("_", "", re.sub("\n", " ", data_text.lower())).split("====")))]
tokenized_documents = [word_tokenize(i) for i in documents]

df = defaultdict(int)
tfs = []
for doc in tokenized_documents:
    d = defaultdict(int)
    for word in doc:
        if word not in d:
            df[word] += 1
        d[word] += 1
    tfs.append({i:math.log(1+j/len(doc), 2) for i,j in d.items()})
idf = {i:math.log(len(documents)/j, 2) for i,j in df.items()}

tfidf_count = defaultdict(int)
tfidf_sum = defaultdict(int)

for i in tfs:
    res = sorted({k:j*idf[k] for k,j in i.items()}.items(), key=lambda x:x[1])
    for j in res:
        tfidf_sum[j[0]] += j[1]
        tfidf_count[j[0]] += 1

tfidf = {i:tfidf_sum[i]/tfidf_count[i] for i in tfidf_count.keys()}

tfidf_sorted = sorted(tfidf.items(), key=lambda i:i[1])
idf_sorted = sorted(idf.items(), key=lambda i:i[1])

word_list = list(filter(lambda x:len(x)>3, set([i[0] for i in tfidf_sorted[-TFIDF_FINAL_INCLUDE:]])))

breakpoint()

