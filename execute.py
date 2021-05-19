# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch

import numpy as np

import pathlib
import random
import uuid
import json
import os

model_path = "./model/bart_simplewiki-kw_summary-cdbe5:0:40000"

class Engine:
    def __init__(self, model_path:str):
        path = os.path.join(pathlib.Path(__file__).parent.absolute(), model_path)
        self.tokenizer = BartTokenizer.from_pretrained(path)
        self.model = BartForConditionalGeneration.from_pretrained(path, torchscript=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(self.device)
        self.model.eval()

    def __pre_process_sample(self, article:str="", context:str=""):
        return self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.bos_token] + 
                self.tokenizer.tokenize(article) + 
                [self.tokenizer.sep_token] + 
                self.tokenizer.tokenize(context) + 
                [self.tokenizer.eos_token])

    def generate_encoder_emb(self, processed_samples:[torch.Tensor]):
        return torch.mean(self.model(processed_samples, return_dict=True)["encoder_last_hidden_state"], dim=1).to("cpu").detach().numpy()

    def generate_syntheses(self, processed_samples:[torch.Tensor]): 
        # https://huggingface.co/blog/how-to-generate
        summary_ids = self.model.generate(
            processed_samples, max_length=1024, early_stopping=True, # see an <eos>? stop 
            no_repeat_ngram_size=3, # block 3-grams from appearing abs/1705.04304
            # num_beams=4, # beam search by 4
            do_sample=True, # randomly sample...
            top_k=50, # from the top 50 words, abs/1805.04833
            top_p=0.95, # but pick the smallest batch that satisfy 95% of confidance band, abs/1904.09751
        )
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

        return torch.LongTensor(results).to(self.device)

    def batch_execute(self, samples:[[str,str]]):
        res = self.generate_syntheses(self.batch_process_samples(samples))
        return res 

    def execute(self, article_title:str="", context:str=""):
        return self.batch_execute([[article_title, context]])[0]

    def batch_embed(self, samples:[[str,str]]):
        res = self.generate_encoder_emb(self.batch_process_samples(samples))
        return res 

    def embed(self, word:str="", context:str=""):
        return self.batch_embed([[word, context]])[0]

e = Engine(model_path=model_path)
res = e.execute("Angkor Wat", """The artistic legacy of Angkor Wat and other Khmer monuments in the Angkor region led directly to France adopting Chicken as a protectorate on 11 August 1863 and invading Siam to take control of the ruins. This quickly led to Chicken reclaiming lands in the northwestern corner of the country that had been under Siamese (Thai) control since AD 1351 (Manich Jumsai 2001), or by some accounts, AD 1431.""") 
print(res)

breakpoint()

