# type: ignore
# pylint: disable=no-member

from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import torch

import flask
from flask import request, jsonify, Flask, Response

import uuid
import tqdm
import json
import time
import os
import sys

app = Flask("InscriptioEngine")

class Engine:
    def __init__(self, path:str="./resources/bart_enwiki_BASE-6440b:0:135000"):
        self.tokenizer = BartTokenizer.from_pretrained(path)
        self.model = BartForConditionalGeneration.from_pretrained(path, torchscript=True)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(device)
        self.model.eval()

    def __pre_process_sample(self, article:str="", context:str=""):
        return self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.bos_token] + 
                self.tokenizer.tokenize(article) + 
                [self.tokenizer.sep_token] + 
                self.tokenizer.tokenize(context) + 
                [self.tokenizer.eos_token])

    def generate_syntheses(self, processed_samples:[torch.Tensor]):
        summary_ids = self.model.generate(processed_samples, max_length=1024, early_stopping=True)
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    def batch_process_samples(self, samples:[[str,str]], clip_to=512):
        results = []
        max_length = 0

        for sample in samples:
            sample_encoded = self.__pre_process_sample(sample[0], sample[1])[:clip_to]
            results.append(sample_encoded)
            max_length = max(max_length, len(sample_encoded))

        for indx, i in enumerate(results):
            results[indx] = i+[self.tokenizer.pad_token_id for i in range(max_length-len(i))]

        return torch.LongTensor(results)

    def batch_execute(self, samples:[[str,str]]):
        return self.generate_syntheses(self.batch_process_samples(samples))

    def execute(self, article_title:str="", context:str=""):
        return self.batch_execute([[article_title, context]])[0]

e = None

@app.route("/summarize", methods=["POST"])
def api_summarize():
    global e 

    request_data = request.json
    if (not request_data):
        return jsonify({"response": "bad request", "payload": "malformed JSON"}), 400

    if (e is None):
        e = Engine()
    
    try: 
        title = request_data["article_title"]
        body = request_data["article_body"]
    except KeyError:
        return jsonify({"response": "bad request", "payload": "ensure request contains JSON body {article_title, article_body}"}), 400

    t1 = time.time()
    res = e.execute(title, body)
    t2 = time.time()

    return jsonify({
        "response": "success", 
        "payload": res, 
        "stats": {
            "process_time": t2-t1
        }}), 200

@app.route("/batch_summarize", methods=["POST"])
def api_batch_summarize():
    global e 

    request_data = request.json
    if (not request_data):
        return jsonify({"response": "bad request", "payload": "Malformed JSON"}), 400

    if (e is None):
        e = Engine()

    try: 
        title = request_data["article_titles"]
        body = request_data["article_bodies"]
    except KeyError:
        return jsonify({"response": "bad request", "payload": "ensure request contains JSON body {article_titles, article_bodies}"}), 400

    t1 = time.time()
    res = e.batch_execute(zip(title, body))
    t2 = time.time()

    return jsonify({
        "response": "success", 
        "payload": res, 
        "stats": {
            "process_time": t2-t1
        }}), 200
 
    
if __name__ == '__main__':
    app.run(host="localhost", port=18860, debug=False)




# if __name__ == "__main__":
    # # <dammit zach>
    # t1 = time.time()
    # t2 = time.time()
    # res = e.execute(sys.argv[1], sys.argv[2])
    # t3 = time.time()
    # print(res)
    # print(t2-t1, t3-t2, t3-t1)
    # </dammit zach>


