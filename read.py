import tqdm
import json
import pickle
from wiki_dump_reader import Cleaner, iterate

titles = []
data = {}
counter = 0
identif = 0

cleaner = Cleaner()

print("Parsing data...")
for title, text in tqdm.tqdm(iterate("./enwiki-20210301-pages-articles-multistream.xml")):
    text = cleaner.clean_text(text)
    cleaned_text, links = cleaner.build_links(text)
    titles.append(title)
    data[title] = {"text": cleaned_text, "links": links}

    counter += 1
    if counter != 0 and counter % 65536 == 0:
        print("Writing parsed datapack ", identif)
        with open(f"./enwiki_data/enwiki-data_{identif}.json", "w") as df:
            json.dump({"titles": titles, "data": data}, df)

        counter = 0
        titles = []
        data = {}
        identif += 1

breakpoint()

# with open("./simplewiki-data.bin", "rb") as df:
    # data = pickle.load(df)

# breakpoint()

