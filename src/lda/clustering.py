import re
import io
import sys
import csv
import json
import glob
import MeCab
import gensim
import pickle
import urllib.request
import pandas as pd
from gensim import corpora, models, similarities

def clustering(dataDir, num_topics):
    corpusFnames = glob.glob(dataDir + "*.mm")
    corpDic = corpora.dictionary.Dictionary.load(dataDir + "corpus.dict")

    corpus0 = corpora.mmcorpus.MmCorpus(corpusFnames[0])
    lda = gensim.models.ldamodel.LdaModel(corpus0, num_topics = num_topics, id2word =corpDic)

    for fname in corpusFnames[1:]:
        corpus = corpora.mmcorpus.MmCorpus(fname)
        lda.update(corpus)

    lda.save(dataDir + "/cur_model.model")

if __name__ == '__main__':
    num_topics = 3
    clustering("../../../Data/Chat-Yojo-Bot/LDA/", num_topics)
