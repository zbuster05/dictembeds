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

model_path = "./model/bart_enwiki-kw_summary-f84c4:ROUTINE::0:60000"

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
                print(f"One sample was a bit long. Truncated to <<... {truncated[-20:]}>> based on {max_length} tokens.")

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
while True:
    word = input("define: ")

    print(e.execute(word.strip(), """The fall of the Roman Empire (476 CE) and the beginning of the European Renaissance in the late fourteenth century roughly bookend the period we call the Middle Ages. Without a dominant centralized power or overarching cultural hub, Europe experienced political and military discord during this time. Its inhabitants retreated into walled cities, fearing marauding pillagers including Vikings, Mongols, Arabs, and Magyars. In return for protection, they submitted to powerful lords and their armies of knights. In their brief, hard lives, few people traveled more than ten miles from the place they were born. The Christian Church remained intact, however, and emerged from the period as a unified and powerful institution. Priests, tucked away in monasteries, kept knowledge alive by collecting and copying religious and secular manuscripts, often adding beautiful drawings or artwork. Social and economic devastation arrived in 1340s, however, when Genoese merchants returning from the Black Sea unwittingly brought with them a rat-borne and highly contagious disease, known as the bubonic plague. In a few short years, it had killed many millions, about one-third of Europe’s population. A different strain, spread by airborne germs, also killed many. Together these two are collectively called the Black Death Entire villages disappeared. A high birth rate, however, coupled with bountiful harvests, meant that the population grew during the next century. By 1450, a newly rejuvenated European society was on the brink of tremendous change. An illustration depicts two bedridden victims, a man and a woman, whose bodies are covered with the swellings characteristic of the Black Death. Another man walks by holding a handful of herbs or flowers. During the Middle Ages, most Europeans lived in small villages that consisted of a manorial house or castle for the lord, a church, and simple homes for the peasants or serfs, who made up about 60 percent of western Europe’s population. Hundreds of these castles and walled cities remain all over Europe. A photograph shows the medieval walled city of Carcassonne. It is surrounded by a high double wall with slots at the top, likely for archers or other defenders to use, and it incorporates several round parapets with narrow window openings. One of the most beautifully preserved medieval walled cities is Carcassonne, France. Notice the use of a double wall. Europe’s feudal society was a mutually supportive system. The lords owned the land; knights gave military service to a lord and carried out his justice; serfs worked the land in return for the protection offered by the lord’s castle or the walls of his city, into which they fled in times of danger from invaders. Much land was communally farmed at first, but as lords became more powerful they extended their ownership and rented land to their subjects. Thus, although they were technically free, serfs were effectively bound to the land they worked, which supported them and their families as well as the lord and all who depended on him. The Catholic Church, the only church in Europe at the time, also owned vast tracts of land and became very wealthy by collecting not only tithes (taxes consisting of 10 percent of annual earnings) but also rents on its lands. A serf’s life was difficult. Women often died in childbirth, and perhaps one-third of children died before the age of five. Without sanitation or medicine, many people perished from diseases we consider inconsequential today; few lived to be older than forty-five. Entire families, usually including grandparents, lived in one- or two-room hovels that were cold, dark, and dirty. A fire was kept lit and was always a danger to the thatched roofs, while its constant smoke affected the inhabitants’ health and eyesight. Most individuals owned no more than two sets of clothing, consisting of a woolen jacket or tunic and linen undergarments, and bathed only when the waters melted in spring. In an agrarian society, the seasons dictate the rhythm of life. Everyone in Europe’s feudal society had a job to do and worked hard. The father was the unquestioned head of the family. Idleness meant hunger."""))

