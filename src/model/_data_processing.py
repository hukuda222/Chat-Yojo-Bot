import io
import sys
import pickle
import random
import collections

import numpy as np


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class PrepareData:
    def __init__(self):
        pass

    ################################################
    def readVocab(self, vocabFile, input_side=False):  # only called when training
        sys.stdout.write("# Vacabulary file name: {}\n".format(vocabFile))

        threshold = 45000 if input_side else 30000

        d = {}
        d.setdefault("<unk>", len(d))  # 0th
        sys.stdout.write("# Vocab: add <unk> | id={}\n".format(d["<unk>"]))
        d.setdefault("<s>", len(d))   # 1th
        sys.stdout.write("# Vocab: add <s>   | id={}\n".format(d["<s>"]))
        d.setdefault("</s>", len(d))  # 2th
        sys.stdout.write("# Vocab: add </s>  | id={}\n".format(d["</s>"]))

        with io.open(vocabFile, "rb") as f:
            vocab_tuple_list = pickle.load(f)
            for word, freq in vocab_tuple_list:
                if word == "<unk>":
                    continue
                elif word == "<s>":
                    continue
                elif word == "</s>":
                    continue
                d.setdefault(word, len(d))
                if len(d) == threshold: break

        sys.stdout.write("# Vocabulary size: {}\n".format(len(d)))

        return d

    ################################################
    def sentence2index(self, sentence, word2indexDict, input_side):
        indexList = [word2indexDict[word] if word in word2indexDict else word2indexDict["<unk>"]
                     for word in sentence.split(" ")]
        out = indexList if input_side else ([word2indexDict["<s>"]] + indexList + [word2indexDict["</s>"]])

        return out

    ################################################
    def makeSentenceLenDict(self, fileName, word2indexDict, input_side=False):
        sys.stdout.write("# Text file name: {}\n".format(fileName))

        d = collections.defaultdict(list) if input_side else {}

        sentenceNum = 0
        sampleNum = 0
        maxLen = 0

        with io.open(fileName, "rb") as f:
            text_list = pickle.load(f)
            for sntNum, snt in enumerate(text_list):
                indexList = self.sentence2index(snt, word2indexDict, input_side)
                sampleNum += len(indexList)
                if input_side:
                    d[len(indexList)].append((sntNum, indexList))
                else:
                    d[sntNum] = indexList
                sentenceNum += 1
                maxLen = max(maxLen, len(indexList))
                # if sentenceNum == 10: break

        sys.stdout.write("# data sent: %10d, sample: %10d, maxlen: %10d\n" % (
                         sentenceNum, sampleNum, maxLen)
        )

        return d

    ################################################
    def makeBatch4Train(self, encSentLenDict, decSentLenDict, batch_size=1, shuffle=True):
        encSentDividedBatch = []
        for length, encSentList in sorted(encSentLenDict.items(), reverse=True):
            # random.shuffle(encSentList)
            iter2 = range(0, len(encSentList), batch_size)
            encSentDividedBatch.extend([encSentList[_:_ + batch_size] for _ in iter2])

        if shuffle:
            sys.stderr.write(("# YES shuffle\n"))
            random.shuffle(encSentDividedBatch)
        else:
            sys.stderr.write(("# NO shuffle: descending order based on encoder sentence length\n"))

        encSentBatch = []
        decSentBatch = []

        for encBatch in encSentDividedBatch[::-1]:
            maxDecoLen = max([len(decSentLenDict[sntNum]) for sntNum, _ in encBatch])
            padLen = lambda sntNum: maxDecoLen - len(decSentLenDict[sntNum])

            encSentBatch.append([np.array(encSent, dtype=np.int32) for _, encSent in encBatch])
            decSentBatch.append([np.array(decSentLenDict[sntNum] + [-1] * padLen(sntNum), dtype=np.int32)
                                 for sntNum, _ in encBatch])

        sys.stdout.write("# Batch number: {}\n".format(len(encSentBatch)))

        return list(zip(encSentBatch, decSentBatch))
