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

    def batch_process_samples(self, samples:[[str,str]]):
        results = []
        max_length = 0

        for sample in samples:
            sample_encoded = self.__pre_process_sample(sample[0], sample[1])
            results.append(sample_encoded)
            max_length = max(max_length, len(sample_encoded))

        for indx, i in enumerate(results):
            results[indx] = i+[self.tokenizer.pad_token_id for i in range(max_length-len(i))]

        return torch.LongTensor(results)

    def execute(self, samples:[[str,str]]):
        return self.generate_syntheses(self.batch_process_samples(samples))


a = time.time()
E = Engine()
b = time.time()
res = E.execute([
    ["Retroviruses", 
    """Viruses that have the ability to intergrate into the chromosomes of the host cell.  

Early Events

* Viruses is uncoated, and uses an enzyme called reverse transcriptase to turn ssRNA to cDNA, and finally into dsDNA
* Then, the enzyme integrase threads the viral dsDNA into the cell's nucleaus
* HIV protease cuts HIV polyproteins into individual parts ready for budding

Late Events

* Proviral region is transcribed slowly whenever ribosome comes across it by the host DNA polymerase II to make viral proteins + replicate the viral genome
* Components are later exported, assembled, and slowly released through budding

To make this happen, the virus needs...

- Reverse Transcriptase
    - Transcript RNA to double-stranded RNA
    - Take double-stranded RNA to turn into DNA
- Integrase
    - Force insert the DNA into the genome of the host cell

And because of the fact that viral DNA is now in cellular DNA, these viruses' DNAs are hard to get rid of.

And this is why we can't cure HIV.

Virus, in this case, spread through cell duplication

* Proviral region on the DNA, every time the ribosome comes across it, makes a new viron
* These components are then assembled, sent, etc. as usual
* Because of the fact that the ribosome needs to, well, come across the bit of DNA for this to work, the virons are made slowly by "trickling out. """],
    ["Mitosis",
    """* Chromesomes line up in equator
* Each chromesome has two chromatid exactly the same
* Microtubials to pull chromesomes appart connected to kinecore, a joint in the chromatid
* Kinetore senses tension, and when it is correct, molecules are sent down the microtubials to send a split signal"""]

])
c = time.time()
breakpoint()

