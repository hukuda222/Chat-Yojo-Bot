import re
import time
import MeCab
import gensim
from gensim import corpora, models, similarities

def print_topics(lda_obj):
    temp = lda.print_topics()
    for line in temp:
        print(line)

if __name__ == '__main__':
    # corpDic = corpora.dictionary.Dictionary.load("../../../Data/Chat-Yojo-Bot/LDA/corpus.dict")
    # print(len(corpDic.items()))

    lda = models.ldamodel.LdaModel.load("../../../Data/Chat-Yojo-Bot/LDA/cur_model.model", mmap = "r")
    lda = models.ldamodel.LdaModel.load("../Data/lda_tmp/cur_model.model", mmap = "r")
    print_topics(lda)
