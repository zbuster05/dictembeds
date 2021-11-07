import json
from tqdm import tqdm
from execute import Engine

MODEL_PATH = "./training/bart_enwiki-kw_summary-2d8df:ROUTINE::1:10000"
e = Engine(model_path=MODEL_PATH)

with open("./validation_wordnet/validation_subset.json", "r") as df:
    raw_validation_data = json.load(df)

len(raw_validation_data)


