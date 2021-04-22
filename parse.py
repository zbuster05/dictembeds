import re

import tqdm
import json
import pickle

data_files = 240

counter = 0
identif = 0

titles = []
keyword_sentences = []
context_sentences = []

for i in tqdm.tqdm(range(data_files)):
    with open(f"./enwiki_data/enwiki-data_{i}.json", "r") as df:
        raw_result = json.load(df) 

    for title in raw_result["titles"]:
        wiki_text = raw_result["data"][title]["text"]
        
        wiki_lines = wiki_text.split("\n")

        if wiki_lines[0].split(" ")[0] == "REDIRECT" or wiki_lines[0].split(" ")[0] == "redirect":
            continue

        wiki_filtered_lines = ""

        try:
            for line in wiki_lines:
                if line[0] == "=":
                    break 
                wiki_filtered_lines = wiki_filtered_lines + " " + line
        except IndexError:
            continue

        wiki_filtered_lines = wiki_filtered_lines.strip()
        wiki_filtered_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', wiki_filtered_lines)

        if (len(wiki_filtered_sentences)>2):
            titles.append(title)
            keyword_sentences.append(wiki_filtered_sentences[0])

            context = ""
            for line in wiki_filtered_sentences[1:]:
                context = context + " " + line
            context_sentences.append(context.strip())

            counter += 1
            if counter != 0 and counter % 65536 == 0:
                with open(f"./enwiki_parsed_titles/enwiki-parsed_{identif}.json", "w") as df:
                    json.dump({"keywords": keyword_sentences, "contexts": context_sentences, "titles": titles}, df)
                counter = 0
                titles = []
                keyword_sentences = []
                context_sentences = []
                identif += 1

