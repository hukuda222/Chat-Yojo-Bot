import requests
import json
import sys
import os
import numpy as np
from requests_oauthlib import OAuth1
from datetime import datetime as dt
from pymongo import MongoClient

sys.path.append("../../../../Data/Chat-Yojo-Bot/")

from Key_and_Token import OAuth

shuffleEpoch = 1000

def followerListGen(collect_Whole_users):
    user_set = set([])
    for data in collect_Whole_users.find({}):
        for user in data["users"]:
            user_set.add(user["id_str"])

    user_list = list(user_set)
    return user_list

def followerStrGen(user_list):
    d = len(user_list)
    np_user_list = np.array(user_list)
    np.random.shuffle(np_user_list)
    shuffled_user_list = np_user_list.tolist()[:5000]
    user_list_str = ", ".join(shuffled_user_list)
    return user_list_str

if __name__ == '__main__':
    print("##########################")
    print("# Start timeline scraing #")
    print("##########################")
    print("\n")

    client_loc = MongoClient("localhost", 27017)
    db_loc_Chat_Yojo_Bot = client_loc.Chat_Yojo_Bot
    collect_Base = db_loc_Chat_Yojo_Bot.Base # document's form is json which taken by twitter REST/STREAM api
    collect_Whole_users = db_loc_Chat_Yojo_Bot.Whole_users

    # for stream tweet scraper.
    url = "https://stream.twitter.com/1.1/statuses/filter.json"
    AObj = OAuth()
    auth = OAuth1(AObj.api_key, AObj.api_secret, AObj.access_token, AObj.access_secret)

    user_list = followerListGen(collect_Whole_users)

    print("------------------------------------------------")
    while True:
        user_list_str = followerStrGen(user_list)
        r = requests.post(url, auth = auth, stream = True, data = {"follow": user_list_str})
        for i, data in enumerate(r.iter_lines()):
            try:
                json_dict = json.loads(data.decode("utf-8"))
                collect_Base.insert_one(json_dict)
                curText = json_dict.get("text")
                if curText is not None:
                    print(curText.encode("cp932", "replace").decode("cp932"))
                else:
                    print("None :/")
                print("------------------------------------------------")
            except:
                print("Something happend :/.")
                print("------------------------------------------------")
            if i > shuffleEpoch:
                print("Let's shuflle :)")
                break
