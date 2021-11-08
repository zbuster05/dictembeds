import nltk
from nltk.corpus import wordnet as wn

def get_word_definition(word):
    try:
        definition = wn.synsets(word+".n.01").definition()
    except(nltk.corpus.reader.wordnet.WordNetError):
        print(word + " was skipped because it doesn't bloody exist")
        definition = None
    return definition

