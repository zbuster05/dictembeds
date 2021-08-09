# type: ignore
# pylint: disable=no-member

from execute import Engine
from nltk.tokenize import word_tokenize
from collections import defaultdict

import math
import re

# model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a2fc9:B_VAL::0:24900:0.8616854640095484"
model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-a5029:ROUTINE::0:30000"
# model_path = "./training/bart_enwiki-kw_summary-cf8cd:ROUTINE::0:20000"

TOP_TFIDF_REMAIN = 50 # number of top words of each doc to include
TOP_CONSIDER_THRESHOLD = 4 # count of top across documents before consideration

with open("./colonialism.txt", "r") as data:
    data_text = data.read()


paragraphs = list(filter(lambda x:(x!='' and len(x)>25), data_text.lower().split("\n")))
tokenized_paragraphs = [word_tokenize(i) for i in paragraphs]

df = defaultdict(int)
tfs = []
for doc in tokenized_paragraphs:
    d = defaultdict(int)
    for word in doc:
        if word not in d:
            df[word] += 1
        d[word] += 1
    tfs.append({i:j/len(doc) for i,j in d.items()})
idf = {i:math.log(len(paragraphs)/j, math.e) for i,j in df.items()}

tfidf_count = defaultdict(int)
for i in tfs:
    res = sorted({k:j*idf[k] for k,j in i.items()}.items(), key=lambda x:x[1])
    for j in res[-TOP_TFIDF_REMAIN:]:
        tfidf_count[j[0]] += 1

tfidf_count = sorted(tfidf_count.items(), key=lambda i:i[1])

breakpoint()



# e = Engine(model_path=model_path)

# @app.route('/predict', methods=['POST'])
# def predict():
    # try:
        # title = request.json["title"]
        # context = request.json["context"]
    # except (KeyError, TypeError):
        # return jsonify({"code": "bad_request", "response": "Bad request. Missing key(s) title, context.", "payload": ""}), 400

    # params = request.json.get("params")

    # try: 
        # if params:
            # result = e.execute(title.strip(), 
                                # re.sub(r"\n", "", context.strip()), 
                                # num_beams=int(params["num_beams"]), 
                                # min_length=int(params["min_length"]), 
                                # no_repeat_ngram_size=int(params["no_repeat_ngram_size"]))
        # else:
            # result = e.execute(title.strip(), re.sub(r"\n", "", context.strip()))
    # except ValueError:
        # return jsonify({"code": "size_overflow", "response": f"Size overflow. The context string is too long and should be less than {e.tokenizer.model_max_length} tokens.", "payload": e.tokenizer.model_max_length}), 418
    # except (KeyError, TypeError):
        # return jsonify({"code": "bad_request", "response": "Bad request. Missing few of key(s) num_beams,  min_length or no_repeat_ngram_size in params.", "payload": ""}), 400

    # return {"code": "success", "response": result}, 200

# if __name__ == "__main__":
    # app.run()


