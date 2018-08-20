import os
import re
import sys
import json
import pickle
import requests
import numpy as np
import pandas as pd
import zenhan as zh
from pymongo import MongoClient
from datetime import datetime as dt
from requests_oauthlib import OAuth1

dataDir = "../../../../Data/Chat-Yojo-Bot/"

def normalizeText(string):
    patternUrl = r"https?://[a-zA-Z0-9/:%#\$&\?\(\)~\.=\+\-_]+"
    patternScreenName = r"@[a-zA-Z0-9/:%#\$&\?\(\)~\.=\+\-_]+"
    patternHashtag = r"#[a-zA-Z0-9/:%#\$&\?\(\)~\.=\+\-_]+"

    rep1str = re.sub(patternUrl, "", string)
    rep2str = re.sub(patternScreenName, "", rep1str)
    rep3str = re.sub(patternHashtag, "", rep2str)
    rep4str = re.sub("(^(\s)*)|((\s)*$)", "", rep3str)
    rep5str = rep4str.replace("\n", "")
    rep6str = zh.z2h(rep5str).lower()

    return rep6str

def extractOrig(collect):
    origDf = pd.DataFrame(np.empty([0, 0]))
    text_list = []
    for data in collect.find({}):
        if "text" in data.keys():
            text_list.append(data["text"])
    origDf["text"] = text_list
    return origDf

def extractConv(collect):
    origDf = pd.DataFrame(np.empty([0, 0]))
    utter_list = []
    rep_list = []
    for data in collect.find({}):
        utter_list.append(data["utter"])
        rep_list.append(data["rep"])
    origDf["utter"] = utter_list
    origDf["rep"] = rep_list
    return origDf

if __name__ == '__main__':
    client_loc = MongoClient("localhost", 27017)
    db_loc_Chat_Yojo_Bot = client_loc.Chat_Yojo_Bot
    collect_Base = db_loc_Chat_Yojo_Bot.Base
    collect_Sample = db_loc_Chat_Yojo_Bot.Sample
    collect_Conversation = db_loc_Chat_Yojo_Bot.Conversation

    BaseDf = extractOrig(collect_Base)
    SampleDf = extractOrig(collect_Sample)
    ConversationDf = extractConv(collect_Conversation)

    BaseDf["text"] = BaseDf["text"].map(normalizeText)
    SampleDf["text"] = SampleDf["text"].map(normalizeText)
    ConversationDf["utter"] = ConversationDf["utter"].map(normalizeText)
    ConversationDf["rep"] = ConversationDf["rep"].map(normalizeText)

    BaseDf = BaseDf.replace("", np.nan).dropna()
    SampleDf = SampleDf.replace("", np.nan).dropna()
    ConversationDf = ConversationDf.replace("", np.nan).dropna()

    BaseFname = dataDir + "Corpus/Base.json"
    SampleFname = dataDir + "Corpus/Sample.json"
    ConversationFname = dataDir + "Corpus/Conversation.json"

    BaseDf.to_json(BaseFname)
    SampleDf.to_json(SampleFname)
    ConversationDf.to_json(ConversationFname)
