import gc
import glob
import MeCab
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import defaultdict

dataDir = "../../../../Data/Chat-Yojo-Bot/"

def checkJap(w):
    return ("あ" <= w and w <= "ん") or ("ア" <= w and w <= "ン") or ("ｱ" <= w and w <= "ﾝ")

def contain(origPhrase):
    wList = list(origPhrase)
    howmany = list(map(lambda x: 1 if checkJap(x) else 0, wList))
    return [sum(howmany), len(howmany)]

def count(origString):
    # global parser
    parser = lambda x: MeCab.Tagger().parse(origString).split(" ")
    utterLine = list(map(contain, parser(origString)))
    whole = sum(list(map(lambda x: x[1], utterLine)))
    child = sum(list(map(lambda x: x[0], utterLine)))
    return [child, whole]

def getRatio(fname):
    Df = pd.read_json(fname, encoding = "utf-8")
    print(fname, len(Df))
    if (len(Df)) == 0: return 0

    Df["base"] = Df.apply(lambda x: x[0] + x[1], axis = 1)
    Df["base"] = Df["base"].map(count)
    Df["child"] = Df["base"].map(lambda x: x[0])
    Df["whole"] = Df["base"].map(lambda x: x[1])

    ratio = sum(list(Df["child"])) / sum(list(Df["whole"]))

    del Df
    gc.collect()

    return ratio

def filterFname(ConversationDfFnameList):
    ratioList = list(map(getRatio, ConversationDfFnameList))
    tfList = list(map(lambda x: True if x >= 0.18 else False, ratioList))
    lastList = [a for (a, b) in zip(ConversationDfFnameList, tfList) if b > 0.18]
    """
    for i, j in zip(ConversationDfFnameList, ratioList):
        print(i, j)

    Df = pd.DataFrame(np.empty([0, 0]))
    Df["ratio"] = ratioList
    Df["ratio"].hist()
    plt.show()
    """
    return lastList

def dumpVocab(Df, column):
    series = Df[column]
    d = defaultdict(lambda: 0)
    for sentence in series:
        for cha in sentence.split(" "):
            d[cha] += 1

    if (column == "utter"):
        fname = dataDir + "Vocab/enc_vocab_AskFM.pickle"
    else:
        fname = dataDir + "Vocab/dec_vocab_AskFM.pickle"

    sorted_tuple_vocab = sorted(d.items(), key=lambda x: x[1], reverse=True)

    with open(fname, 'wb') as f:
        pickle.dump(sorted_tuple_vocab, f)

def dumpText(Df, column):
    if (column == "utter"):
        fname = dataDir + "Corpus/enc_text_AskFM.pickle"
    else:
        fname = dataDir + "Corpus/dec_text_AskFM.pickle"

    texts = list(Df[column])
    with open(fname, 'wb') as f:
        pickle.dump(texts, f)

if __name__ == "__main__":
    jsons = dataDir + "AskFM/*.json"
    ConversationDfFnameList = glob.glob(jsons)
    ConversationDfFnameList = filterFname(ConversationDfFnameList)

    Df = pd.concat([pd.read_json(i, encoding = "utf-8") for i in ConversationDfFnameList], ignore_index = True)

    mecab = MeCab.Tagger("-Owakati") # be sure explicitlly state which dictionary to use, in here default MeCab dictionary is configured as neologd
    mecab.parse("") # add to avoid internal bug
    Df["utter"] = Df["utter"].map(lambda x: mecab.parse(x.strip()).strip())
    Df["rep"] = Df["rep"].map(lambda x: mecab.parse(x.strip()).strip())

    dumpVocab(Df, 'utter')
    dumpVocab(Df, 'rep')

    dumpText(Df, 'utter')
    dumpText(Df, 'rep')
