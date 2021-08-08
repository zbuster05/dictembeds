import re
import json
from tqdm import tqdm
from wiki_dump_reader import Cleaner, iterate

from nltk.tokenize import sent_tokenize

database = []
index = {}

prefix = "enwiki"

cleaner = Cleaner()
for title, text in tqdm(iterate(f"./source/{prefix}-latest-pages-articles.xml"), total=21181268):
# for title, text in tqdm(iterate(f"./source/{prefix}-latest-pages-articles.xml"), total=346229):
    text = cleaner.clean_text(text)
    cleaned_text, links = cleaner.build_links(text)
    
    abstract = []
    passage = []

    abstracting = True
    clean_text = re.sub(r"__NOTOC__", "", cleaned_text)

    for i in clean_text.split("\n"):
        try: 
            if (i[0] == "="):
                abstracting = False
            else:
                if abstracting:
                    abstract.append(i)
                else:
                    passage.append(i+"\n")
        except IndexError:
            continue;

    if (len(abstract) < 4) or (len(passage) < 20):
        continue

    abstract_text = " ".join(abstract)
    passage_text = " ".join(passage)

    linkdb = []
    for i in links:
        linkdb.append(i["link"])

    abstract_splits = sent_tokenize(abstract_text)
    front_raw = abstract_splits.pop(0)

    passage_splits = sent_tokenize(passage_text)
    back_raw = passage_splits.pop()

    if len(abstract_splits) < 5 or len(passage_splits) < 50:
        continue

    # things in the parens often suck.
    front = re.sub("  ", " ", re.sub(r"\(.*?\)", "", front_raw)) 

    abstract_text = abstract_text.replace(front_raw, "").strip()
    passage_text = passage_text.replace(back_raw, "").strip()

    database.append({"title": title, "context": abstract_text, "full_context": passage_text, "target": front, "links": linkdb, "oncontext": True})
    index[title] = front

ldatabase = []
i = 0
for item in tqdm(database, total=len(database)):
    ldatabase.append(item)

    if len(ldatabase) > 53760:
        with open(f"./data/{prefix}-parsed-long-oc-MD{i}.json", "w") as df:
            df.write(json.dumps(ldatabase))
            ldatabase = []
            i += 1

with open(f"./data/{prefix}-parsed-long-oc-MD{i}.json", "w") as df:
    df.write(json.dumps(database))
    ldatabase = []
    i += 1

ldatabase = []
i = 0
for item in tqdm(database, total=len(database)):
    try: 
        for link in item["links"]:
            try: 
                ldatabase.append({"title": link, "context": item["context"], "full_context": item["full_context"], "target": index[link], "oncontext": False})
            except KeyError:
                continue
    except KeyError:
        continue

    if len(ldatabase) > 53760:
        with open(f"./data/{prefix}-parsed-long-oc-OC{i}.json", "w") as df:
            df.write(json.dumps(ldatabase))
            ldatabase = []
            i += 1

with open(f"./data/{prefix}-parsed-long-oc-OC{i}.json", "w") as df:
    df.write(json.dumps(ldatabase))
    ldatabase = []
    i += 1


breakpoint()



