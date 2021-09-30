from rouge_score import rouge_scorer
import json
import tqdm

from execute import Engine

validation_data_originals = []
print("Caching originals data...")
for i in tqdm.tqdm(range(5,6)):
    filename = f"./data/enwiki-parsed-long-oc-MD{i}.json"
    with open(filename, "r") as df:
        validation_data_originals = validation_data_originals + json.load(df)

validation_data_oc = []
print("Caching OC data...")
for i in tqdm.tqdm(range(5,6)):
    filename = f"./data/enwiki-parsed-long-oc-OC{i}.json"
    with open(filename, "r") as df:
        validation_data_oc = validation_data_oc + json.load(df)


model_path = "./training/bart_enwiki-kw_summary-2d8df:ROUTINE::1:10000"
e = Engine(model_path=model_path)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


rouge1_prec = []
rouge1_recc = []
rouge1_fm = []

rougel_prec = []
rougel_recc = []
rougel_fm = []

for i in tqdm.tqdm(validation_data_originals[:500]):
    results = scorer.score(i["target"], e.execute(i["title"], i["context"], num_beams=2, min_length=0))

    rouge1_prec.append(results["rouge1"].precision)
    rouge1_recc.append(results["rouge1"].recall)
    rouge1_fm.append(results["rouge1"].fmeasure)

    rougel_prec.append(results["rougeL"].precision)
    rougel_recc.append(results["rougeL"].recall)
    rougel_fm.append(results["rougeL"].fmeasure)

for i in tqdm.tqdm(validation_data_oc[:500]):
    results = scorer.score(i["target"], e.execute(i["title"], i["context"], num_beams=2, min_length=0))

    rouge1_prec.append(results["rouge1"].precision)
    rouge1_recc.append(results["rouge1"].recall)
    rouge1_fm.append(results["rouge1"].fmeasure)

    rougel_prec.append(results["rougeL"].precision)
    rougel_recc.append(results["rougeL"].recall)
    rougel_fm.append(results["rougeL"].fmeasure)



sum(rouge1_prec)/len(rouge1_prec) # 0.535419
sum(rouge1_recc)/len(rouge1_recc) # 0.395328
sum(rouge1_fm)/len(rouge1_fm)     # 0.434306

sum(rougel_prec)/len(rougel_prec) # 0.497698
sum(rougel_recc)/len(rougel_recc) # 0.368588
sum(rougel_fm)/len(rougel_fm)     # 0.404528

