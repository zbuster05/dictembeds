# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS


import numpy as np

import pathlib
import random
import uuid
import json
import os

app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app)

model_path = "./training/bart_enwiki-kw_summary-12944:ROUTINE::0:30000"

class Engine:
    def __init__(self, model_path:str):
        path = os.path.join(pathlib.Path(__file__).parent.absolute(), model_path)
        self.tokenizer = BartTokenizer.from_pretrained(path)
        self.model = BartForConditionalGeneration.from_pretrained(path)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(self.device)
        self.model.eval()

    def __pre_process_sample(self, article:str="", context:str=""):
        return self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.bos_token] + 
                self.tokenizer.tokenize(article.lower()) + 
                [self.tokenizer.mask_token] + 
                self.tokenizer.tokenize(context) + 
                [self.tokenizer.eos_token])

    def final_decoder_hidden_mean(self, processed_samples:[torch.Tensor]):
        return np.mean((self.model(processed_samples, return_dict=True, output_hidden_states=True))["decoder_hidden_states"][-1].to("cpu").detach().numpy(), axis=0)

    def generate_syntheses(self, processed_samples:[torch.Tensor]): 
        # https://huggingface.co/blog/how-to-generate
        summary_ids = self.model.generate(
            processed_samples,
            decoder_start_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=5, # block 3-grams from appearing abs/1705.04304
            # num_beams=5,
            max_length=1000,
            do_sample=True,
            top_p = 0.90,
            top_k = 20
        )
        return [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    def batch_process_samples(self, samples:[[str,str]]):
        results = []
        max_length = self.tokenizer.model_max_length

        for sample in samples:
            sample_encoded = self.__pre_process_sample(sample[0], sample[1])

            if len(sample_encoded) > max_length:
                sample_encoded = self.__pre_process_sample(sample[0], sample[1])[:max_length]
                truncated = self.tokenizer.decode(sample_encoded)
                raise ValueError(f"One sample was a bit long. Truncated to <<... {truncated[-20:]}>> based on {max_length} tokens.")

            results.append(sample_encoded)
            max_length = max(max_length, len(sample_encoded))

        for indx, i in enumerate(results):
            results[indx] = i+[self.tokenizer.pad_token_id for i in range(max_length-len(i))]

        return torch.LongTensor(results).to(self.device)

    def batch_execute(self, samples:[[str,str]]):
        res = self.generate_syntheses(self.batch_process_samples(samples))
        return res 

    def execute(self, article_title:str="", context:str=""):
        return self.batch_execute([[article_title, context]])[0]

e = Engine(model_path=model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        title = request.json["title"]
        context = request.json["context"]
    except (KeyError, TypeError):
        return jsonify({"code": "bad_request", "response": "Bad request. Missing key(s) title, context.", "payload": ""}), 400

    try: 
        result = e.execute(title.strip(), context.strip())
    except ValueError:
        return jsonify({"code": "size_overflow", "response": f"Size overflow. The context string is too long and should be less than {e.tokenizer.model_max_length} tokens.", "payload": e.tokenizer.model_max_length}), 418

    return {"code": "success", "response": result}, 200

if __name__ == "__main__":
    app.run()


