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

model_path = "./model/bart_enwiki-kw_summary-04fd6:0:30000"

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
                [self.tokenizer.sep_token] + 
                self.tokenizer.tokenize(context) + 
                [self.tokenizer.eos_token])

    def final_decoder_hidden_mean(self, processed_samples:[torch.Tensor]):
        return np.mean((self.model(processed_samples, return_dict=True, output_hidden_states=True))["decoder_hidden_states"][-1].to("cpu").detach().numpy(), axis=0)

    def generate_syntheses(self, processed_samples:[torch.Tensor]): 
        # https://huggingface.co/blog/how-to-generate
        summary_ids = self.model.generate(
            processed_samples,
            no_repeat_ngram_size=3, # block 3-grams from appearing abs/1705.04304
            # num_beams=2,
            do_sample=True,
            top_p = 0.90
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

    # def generate_encoder_emb(self, processed_samples:[torch.Tensor]):
        # return torch.mean(self.model(processed_samples, return_dict=True)["encoder_last_hidden_state"], dim=1).to("cpu").detach().numpy()

#     def batch_embed(self, samples:[[str,str]]):
        # res = self.generate_encoder_emb(self.batch_process_samples(samples))
        # return res 

    # def embed(self, word:str="", context:str=""):
        # return self.batch_embed([[word, context]])[0]

e = Engine(model_path=model_path)
while True:
    # print(e.execute("transformer", """a transformer has one of those large arms that kill people"""))
#     print(e.execute("transformer", """a transformer is a language model that is super duper useful"""))
    # print(e.execute("transformer", """a transformer kills people using its large arms"""))
    # print(e.execute("language model", """a transformer is a language model that is super duper useful"""))

    # breakpoint()

    word = input("define: ")
    print(e.execute(word.strip(), """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.""") )
    # print(e.execute(word.strip(), """"About 12,000 years ago, humans crossed an important threshold when they began to experiment with agriculture. It quickly became clear that cultivation of crops provided a larger and more reliable food supply than foraging. Groups that turned to agriculture experienced rapid population growth, and they settled into permanent communities. Some of these developed into cities and became the world’s first complex societies. The term complex society refers to a form of large-scale social organization in which productive agricultural economies produced surplus food. That surplus allowed some people to devote their time to specialized tasks, other than food production, and to congregate in urban settlements. During the centuries from 3500 to 500 B.C.E., complex societies arose independently in several regions of the world, including Mesopotamia, Egypt, northern India, China, Mesoamerica, and the central Andean region of South America. Each established political authorities, built states with formal governmental institutions, collected surplus agricultural production in the form of taxes or tribute, and redistributed wealth. Complex societies also traded with other peoples, and they often sought to extend their authority to surrounding territories."Complex societies were able to generate and preserve much more wealth than smaller societies. When bequeathed to heirs and held within particular families, this accumulated wealth became the foundation for social distinctions. These societies developed different kinds of social distinctions, but all recognized several classes of people, including ruling elites, common people, and slaves.All early complex societies also created sophisticated cultural traditions. Most of them either invented or borrowed a system of writing, which quickly came to be used to construct traditions of literature, learning, and reflection. All the complex societies organized systems of formal education that introduced intellectual elites to skills such as writing and astronomical observation deemed necessary for their societies’ survival. In addition, all of these societies explored the nature of humanity, the world, and the gods. Although all the early complex societies shared some common features, each nevertheless developed distinct cultural, political, social, and economic traditions of its own. These distinctions were based, at least initially, on geographical differences and the differing availability of resources. For instance, the absence or presence of large supplies of freshwater, of river or ocean transport, of mountains or desert, or of large draft animals or domesticable plants helped"""))

