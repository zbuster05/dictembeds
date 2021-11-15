import json
from tqdm import tqdm
from execute import Engine
from rouge_score import rouge_scorer

MODEL_PATH = "./training/bart_enwiki-kw_summary-2d8df:ROUTINE::1:10000"
e = Engine(model_path=MODEL_PATH)

with open("./validation_wordnet/validation_subset.json", "r") as df:
    raw_validation_data = json.load(df)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
pairs = [[(i["title"], i["context"]), i["target"]] for i in raw_validation_data]

collected_pairs = []
for sample in pairs:
    word = sample[0][0]
    #definition = CallZachsWonderfulFunction(word.lower())
    definition = ''

    if definition:
        collected_pairs.append([sample[0], (sample[1], definition)])

