# type: ignore
# pylint: disable=no-member

from execute import Engine
from nltk.tokenize import word_tokenize,sent_tokenize
from collections import defaultdict
import tqdm

import json
import math
import re

# model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a2fc9:B_VAL::0:24900:0.8616854640095484"
# model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a5029:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-cf8cd:ROUTINE::0:20000"
# model_path = "./training/bart_enwiki-kw_summary-3dee1:B_VAL::0:47200:1.8260153889656068"
# model_path = "./training/bart_enwiki-kw_summary-e2f01:B_VAL::0:59800:1.1983055472373962"
# model_path = "./training/bart_enwiki-kw_summary-07e7d:ROUTINE::0:50000"
# model_path = "./training/bart_enwiki-kw_summary-07e7d:ROUTINE::0:50000"
model_path = "./training/bart_enwiki-kw_summary-2d8df:ROUTINE::1:10000"

TFIDF_FINAL_INCLUDE = 100 # "important" words to include
TOTAL_CONTEXT_SIZE = 20 # size of context to give to model for each term
OCCURENCE_CONTEXT = 1 # size of context around each occurance to have
MIN_LENGTH = 0 # minimum length of summaries

print("loading corpus...")

with open("./amercianrev.txt", "r") as data:
    data_text = data.read()

print("tokenizing corpus...")
documents = [i.strip() for i in list(filter(lambda x:(x!='' and len(x)>1000), re.sub("_", "", re.sub("\n", " ", data_text.lower())).split("====")))]
tokenized_documents = [word_tokenize(i) for i in documents]
sentences = [i for j in [sent_tokenize(i) for i in documents] for i in j]

identify = input("Auto-identify? (y/N) ")
if identify.lower() == 'y':
    print("calculating TFIDF...")
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

    word_list = list(filter(lambda x:len(x)>3 and not sum(c.isdigit() for c in x) > 0.5*len(x), set([i[0] for i in tfidf_sorted])))[-TFIDF_FINAL_INCLUDE:]
else:
    word_list = []
    word = ''
    while word.lower() != "q":
        if len(word) > 1:
            word_list.append(word)
        word = input("Word to define (q for quit): ").strip()

contexts = {}
max_count = defaultdict(int)

print("creating contexts...")
for word in tqdm.tqdm(word_list):
    word_context = []
    for i in range(len(sentences)):
        if (len(word_context) >= TOTAL_CONTEXT_SIZE):
            break
        if word in sentences[i]:
            word_context = word_context + sentences[max(i-OCCURENCE_CONTEXT,0):i+OCCURENCE_CONTEXT]
    word_context = list(set(word_context))
    word_context = word_context[:TOTAL_CONTEXT_SIZE]
    contexts[word] = "".join([i.strip()+" " for i in word_context])

print("instantiating model...")
e = Engine(model_path=model_path)

glossary = {}

print("running predictions...")
for word, context in tqdm.tqdm(contexts.items()):
    result = e.execute(word.strip().lower(), context[:1024], 
                       num_beams=2, min_length=MIN_LENGTH, 
                       no_repeat_ngram_size=2)

    if result != "<CND>" and result != "<>":
        glossary[word] = result

with open("./glossary.json", "w") as df:
    df.write(json.dumps(glossary))
breakpoint()

