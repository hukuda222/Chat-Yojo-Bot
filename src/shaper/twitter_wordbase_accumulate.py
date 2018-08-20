import gc
import glob
import MeCab
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import defaultdict

dataDir = "../../../../Data/Chat-Yojo-Bot/"

def dumpVocab(Df, column):
    series = Df[column]
    d = defaultdict(lambda: 0)
    for sentence in series:
        for cha in sentence.split(" "):
            d[cha] += 1

    if (column == "utter"):
        fname = dataDir + "Vocab/enc_vocab_Twitter.pickle"
    else:
        fname = dataDir + "Vocab/dec_vocab_Twitter.pickle"

    sorted_tuple_vocab = sorted(d.items(), key=lambda x: x[1], reverse=True)

    with open(fname, 'wb') as f:
        pickle.dump(sorted_tuple_vocab, f)

def dumpText(Df, column):
    if (column == "utter"):
        fname = dataDir + "Corpus/enc_text_Twitter.pickle"
    else:
        fname = dataDir + "Corpus/dec_text_Twitter.pickle"

    texts = list(Df[column])
    with open(fname, 'wb') as f:
        pickle.dump(texts, f)

if __name__ == "__main__":
    # sampleFname = "../../../../Data/Chat-Yojo-Bot/Corpus/Sample.json"
    # baseFname = "../../../../Data/Chat-Yojo-Bot/Corpus/Base.json"
    conversationFname = dataDir + "Corpus/Conversation.json"

    fnameList = [conversationFname]

    Df = pd.concat([pd.read_json(i, encoding = "utf-8") for i in fnameList], ignore_index = True)

    mecab = MeCab.Tagger("-Owakati") # be sure explicitlly state which dictionary to use, in here default MeCab dictionary is configured as neologd
    mecab.parse("") # add to avoid internal bug
    Df["utter"] = Df["utter"].map(lambda x: mecab.parse(x.strip()).strip())
    Df["rep"] = Df["rep"].map(lambda x: mecab.parse(x.strip()).strip())

    dumpVocab(Df, 'utter')
    dumpVocab(Df, 'rep')

    dumpText(Df, 'utter')
    dumpText(Df, 'rep')
