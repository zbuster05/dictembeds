import nltk
from nltk.corpus import wordnet as wn

def get_word_definition(word):
    definitions = []
    try:
        for definition in wn.synsets(word):
            definitions.append(definition.definition())
    except(nltk.corpus.reader.wordnet.WordNetError):
        # print(word + " was skipped because it doesn't bloody exist")
        definition = None
    return definitions

