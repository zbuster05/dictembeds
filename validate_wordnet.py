import json
from tqdm import tqdm
from execute import Engine
from rouge_score import rouge_scorer

from wordnet import wordnet
import csv


MODEL_PATH = "./training/bart_enwiki-kw_summary-2d8df:ROUTINE::1:10000"
e = Engine(model_path=MODEL_PATH)

with open("./validation_wordnet/validation_subset.json", "r") as df:
    raw_validation_data = json.load(df)

pairs = [[(i["title"], i["context"]), i["target"]] for i in raw_validation_data]

collected_pairs = []
for sample in pairs:
    word = sample[0][0]
    definitions = wordnet.get_word_definition(word)

    if len(definitions) > 0:
        collected_pairs.append([sample[0], (sample[1], definitions)])

print(f"Validating upon {len(collected_pairs)} collected pairs!")

rouge1_prec = []
rouge1_recc = []
rouge1_fm = []

rougel_prec = []
rougel_recc = []
rougel_fm = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

for sample in tqdm(collected_pairs):
    # Get output and compare with results
    output = e.execute(sample[0][0], sample[0][0], num_beams=2, min_length=10)
    results = [scorer.score(i, output) for i in sample[1][1]]

    # get synsyet with highest scoreso
    # rouge1
    results_rouge1 = [i["rouge1"] for i in results]
    results_rouge1.sort(key=lambda x: (x.precision+x.recall+x.fmeasure)/3)
    rouge1 = results_rouge1[-1]

    # rougel
    results_rougel = [i["rougeL"] for i in results]
    results_rougel.sort(key=lambda x: (x.precision+x.recall+x.fmeasure)/3)
    rougel = results_rouge1[-1]

    # append results
    rouge1_prec.append(rouge1.precision)
    rouge1_recc.append(rouge1.recall)
    rouge1_fm.append(rouge1.fmeasure)

    rougel_prec.append(rougel.precision)
    rougel_recc.append(rougel.recall)
    rougel_fm.append(rougel.fmeasure)

sum(rouge1_prec)/len(rouge1_prec) # 0.17263
sum(rouge1_recc)/len(rouge1_recc) # 0.22024
sum(rouge1_fm)/len(rouge1_fm)     # 0.17347
                                    
sum(rougel_prec)/len(rougel_prec) # 0.17264
sum(rougel_recc)/len(rougel_recc) # 0.22024
sum(rougel_fm)/len(rougel_fm)     # 0.17347


    # title.append(i["title"])
#     context.append(i["context"])
#     model_output.append(output)
#     desired_output.append(i["target"])

#     rouge1_prec.append(results["rouge1"].precision)
#     rouge1_recc.append(results["rouge1"].recall)
#     rouge1_fm.append(results["rouge1"].fmeasure)

#     rougel_prec.append(results["rougeL"].precision)
#     rougel_recc.append(results["rougeL"].recall)
#     rougel_fm.append(results["rougeL"].fmeasure)




