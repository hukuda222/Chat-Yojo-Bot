import re
import io
import sys
import csv
import json
import MeCab
import gensim
import pickle
import urllib.request
import pandas as pd
from gensim import corpora, models, similarities

p = MeCab.Tagger("mecabrc")

def stopWordList():
    slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    slothlib_file = urllib.request.urlopen(slothlib_path)
    slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
    slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss == u'']
    slothlib_stopwords.extend(["ので", "けど", "って", "する", "。", "、", "です",
                                "ある", "ます", "…", "この", "ない", "てる", "れる",
                                "やる","いる", "ござる", "なる", "ー", "たい", "なんか",
                                "たい", "とか", "あの", "まで", "なんか", "せる", "その",
                                "という", "だけ", "くださる", "いい", "なぁ", "じゃ", "でも"])
    return slothlib_stopwords

stopwordList = stopWordList()

def dropWords(po):
    po = re.sub(r'[".¥,¥@]+', '', po)
    po = re.sub(r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]', '', po)
    po = re.sub(r'[\r|\t|]', '', po)
    po = re.sub(r'[!-~]', '', po)
    po = re.sub(r'[︰-＠]', '', po)
    po = po.replace('\n', ' ')
    po = po.replace('　', ' ')
    po.replace('RT', '')
    po.replace('rt', '')
    return po

def executeWords(po):
    local_docs = list()
    node = p.parseToNode(po)

    while node:
        pos = node.feature.split(",")[0]
        word = node.surface
        if (pos == "名詞" or pos == "形容詞") and word not in stopwordList:
            local_docs.append(word)
        node = node.next

    return local_docs

def save(docs, iterter, output_path, dictionary):
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    dictionary.save(output_path + '/corpus.dict')
    corpora.MmCorpus.serialize(output_path + '/cop' + str(iterter) + '.mm', corpus)

def json_to_copdict(f, iterter, output_path, dictionary):
    docs = list()
    all_text=list()
    Df = pd.read_json(f, encoding = "utf-8")

    for po in Df:
        all_text.extend(Df[po])

    for (j, origString) in enumerate(all_text):
        if(len(origString) >= 1):
            droppedString = dropWords(origString)
            local_docs = executeWords(droppedString)
            local_docs = list(set(local_docs).difference(set(stopwordList)))
            dictionary.add_documents([local_docs])
            docs.append(local_docs)

        if j % 100000 == 0 and j > 1:
            print(j)
            save(docs, iterter, output_path, dictionary)
            docs = list()
            iterter += 1

    save(docs, iterter, output_path, dictionary)
    iterter += 1
    docs = list()
    return iterter

def json_to_model(json_list, output_path):
    iterter = 0
    dictionary = corpora.Dictionary([])
    for json_file in json_list:
        iterter = json_to_copdict(json_file, iterter, output_path, dictionary)

if __name__ == '__main__':
    json_to_model(["../../../Data/Chat-Yojo-Bot/Corpus/Sample.json", "../../../Data/Chat-Yojo-Bot/Corpus/Base.json"], "../../../Data/Chat-Yojo-Bot/LDA")
