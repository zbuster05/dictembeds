# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch

import numpy as np

import pathlib
import random
import uuid
import json
import re
import os


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

    def generate_syntheses(self, processed_samples:[torch.Tensor], num_beams:int=5, min_length:int=10, no_repeat_ngram_size:int=3): 
        # https://huggingface.co/blog/how-to-generate
        summary_ids = self.model.generate(
            processed_samples,
            decoder_start_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size, # block 3-grams from appearing abs/1705.04304
            num_beams=num_beams,
            max_length=1000,
            min_length=min_length,
            length_penalty = 0.8,
            repetition_penalty = 1.2
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

    def batch_execute(self, samples:[[str,str]], num_beams:int=5, min_length:int=10, no_repeat_ngram_size:int=3):
        res = self.generate_syntheses(self.batch_process_samples(samples), num_beams, min_length, no_repeat_ngram_size)
        return res 

    def execute(self, article_title:str="", context:str="", num_beams:int=5, min_length:int=10, no_repeat_ngram_size:int=3):
        return self.batch_execute([[article_title, context]], num_beams, min_length, no_repeat_ngram_size)[0]

