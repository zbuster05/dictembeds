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

model_path = "./model/bart_enwiki-kw_summary-56421:ROUTINE::0:40000"

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
            num_beams=5,
            # do_sample=True,
            # top_p = 0.90,
            # top_k = 20
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
    print(e.execute(word.strip(), """The Spanish established the first European settlements in the Americas, beginning in the Caribbean and, by 1600, extending throughout Central and South America. Thousands of Spaniards flocked to the Americas seeking wealth and status. The most famous of these Spanish adventurers are Christopher Columbus (who, though Italian himself, explored on behalf of the Spanish monarchs), Hernán Cortés, and Francisco Pizarro.  The history of Spanish exploration begins with the history of Spain itself. During the fifteenth century, Spain hoped to gain advantage over its rival, Portugal. The marriage of Ferdinand of Aragon and Isabella of Castile in 1469 unified Catholic Spain and began the process of building a nation that could compete for worldwide power. Since the 700s, much of Spain had been under Islamic rule, and King Ferdinand II and Queen Isabella I, arch-defenders of the Catholic Church against Islam, were determined to defeat the Muslims in Granada, the last Islamic stronghold in Spain. In 1492, they completed the Reconquista: the centuries-long Christian conquest of the Iberian Peninsula. The Reconquista marked another step forward in the process of making Spain an imperial power, and Ferdinand and Isabella were now ready to look further afield.  Their goals were to expand Catholicism and to gain a commercial advantage over Portugal. To those ends, Ferdinand and Isabella sponsored extensive Atlantic exploration. Spain’s most famous explorer, Christopher Columbus, was actually from Genoa, Italy. He believed that, using calculations based on other mariners’ journeys, he could chart a westward route to India, which could be used to expand European trade and spread Christianity. Starting in 1485, he approached Genoese, Venetian, Portuguese, English, and Spanish monarchs, asking for ships and funding to explore this westward route. All those he petitioned—including Ferdinand and Isabella at first—rebuffed him; their nautical experts all concurred that Columbus’s estimates of the width of the Atlantic Ocean were far too low. However, after three years of entreaties, and, more important, the completion of the Reconquista, Ferdinand and Isabella agreed to finance Columbus’s expedition in 1492, supplying him with three ships: the Nina, the Pinta, and the Santa Maria. The Spanish monarchs knew that Portuguese mariners had reached the southern tip of Africa and sailed the Indian Ocean. They understood that the Portuguese would soon reach Asia and, in this competitive race to reach the Far East, the Spanish rulers decided to act.  Columbus held erroneous views that shaped his thinking about what he would encounter as he sailed west. He believed the earth to be much smaller than its actual size and, since he did not know of the existence of the Americas, he fully expected to land in Asia. On October 12, 1492, however, he made landfall on an island in the Bahamas. He then sailed to an island he named Hispaniola (present-day Dominican Republic and Haiti) (Figure 2.4). Believing he had landed in the East Indies, Columbus called the native Taínos he found there “Indios,” giving rise to the term “Indian” for any native people of the New World. Upon Columbus’s return to Spain, the Spanish crown bestowed on him the title of Admiral of the Ocean Sea and named him governor and viceroy of the lands he had discovered. As a devoted Catholic, Columbus had agreed with Ferdinand and Isabella prior to sailing west that part of the expected wealth from his voyage would be used to continue the fight against Islam.""") )
    # print(e.execute(word.strip(), """"About 12,000 years ago, humans crossed an important threshold when they began to experiment with agriculture. It quickly became clear that cultivation of crops provided a larger and more reliable food supply than foraging. Groups that turned to agriculture experienced rapid population growth, and they settled into permanent communities. Some of these developed into cities and became the world’s first complex societies. The term complex society refers to a form of large-scale social organization in which productive agricultural economies produced surplus food. That surplus allowed some people to devote their time to specialized tasks, other than food production, and to congregate in urban settlements. During the centuries from 3500 to 500 B.C.E., complex societies arose independently in several regions of the world, including Mesopotamia, Egypt, northern India, China, Mesoamerica, and the central Andean region of South America. Each established political authorities, built states with formal governmental institutions, collected surplus agricultural production in the form of taxes or tribute, and redistributed wealth. Complex societies also traded with other peoples, and they often sought to extend their authority to surrounding territories."Complex societies were able to generate and preserve much more wealth than smaller societies. When bequeathed to heirs and held within particular families, this accumulated wealth became the foundation for social distinctions. These societies developed different kinds of social distinctions, but all recognized several classes of people, including ruling elites, common people, and slaves.All early complex societies also created sophisticated cultural traditions. Most of them either invented or borrowed a system of writing, which quickly came to be used to construct traditions of literature, learning, and reflection. All the complex societies organized systems of formal education that introduced intellectual elites to skills such as writing and astronomical observation deemed necessary for their societies’ survival. In addition, all of these societies explored the nature of humanity, the world, and the gods. Although all the early complex societies shared some common features, each nevertheless developed distinct cultural, political, social, and economic traditions of its own. These distinctions were based, at least initially, on geographical differences and the differing availability of resources. For instance, the absence or presence of large supplies of freshwater, of river or ocean transport, of mountains or desert, or of large draft animals or domesticable plants helped"""))

