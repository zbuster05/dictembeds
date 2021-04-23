# type: ignore
# pylint: disable=no-member

from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
import torch

import uuid
import tqdm
import json
import time
import os

class Engine:
    def __init__(self, path:str="./training/bart_enwiki_BASE-6c279:0:400000"):
        self.tokenizer = BartTokenizer.from_pretrained(path)
        self.model = BartForConditionalGeneration.from_pretrained(path)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

    def __pre_process_sample(self, article:str="", context:str=""):
        return self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.bos_token] + 
                self.tokenizer.tokenize(article) + 
                [self.tokenizer.sep_token] + 
                self.tokenizer.tokenize(context) + 
                [self.tokenizer.eos_token])

    def generate_syntheses(self, processed_samples:[torch.Tensor]):
        summary_ids = self.model.generate(processed_samples, num_beams=4, early_stopping=True)
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

if __name__ == "__main__":
    # dammit zach
    e = Engine()
    res = e.execute("Alan Turing", """Alan turing was a mathematition and chyptographer who worked to break German encryption and ciphers during WWI and established a series of theories on algorithmic computability.

## Work at Combridge Regarding Computability
Turing crated proof that a "universal computing machine", as defined by specific parametres, could compute any mathematical operation as long as it is algorithmically representable.

Furthermore, he showed that there are no solutions to the base-case decision problem (finding the provability of an theorem based on only axioms in O(1) time) because his "computing machines" could not have a finite state by when they halt.

## Hut8
Turing invented a system to decode German communication ("Enigma") with a rotating weights sytem in the Hut8 program at Bletchley Park; however, Turing did not support the method by which the US navy decided to execute upon codebreaking. His codebreaking efforts, by estimate, shaved 2 years from the war.

He also assisted in creating secured speech systems for the navy.""")
    print(res)
    breakpoint()


