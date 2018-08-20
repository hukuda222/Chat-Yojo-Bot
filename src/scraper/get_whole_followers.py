import requests
import pickle
import json
import time
import sys
import os
from requests_oauthlib import OAuth1
import datetime
from pymongo import MongoClient
from pprint import pprint

sys.path.append("../../../Data/Chat-Yojo-Bot/")

from Key_and_Token import OAuth

follower_limitation = 400

def now_unix_time():
    return time.mktime(datetime.datetime.now().timetuple())

def get_partial_followers_user_id(url, auth, user_id, cur_cusor, count, collect_Whole_users):
    while True:
        r = requests.get(url, auth = auth, params = {"user_id": user_id, "cursor": cur_cusor, "count": count})
        limit = r.headers['x-rate-limit-remaining'] if 'x-rate-limit-remaining' in r.headers else 0
        print(limit, ": limit")
        reset = float(r.headers['x-rate-limit-reset']) if 'x-rate-limit-reset' in r.headers else 0
        print(reset, ": reset")

        if int(limit) == 0:
            diff_sec = reset - now_unix_time()
            print("Sleep %d sec." % (diff_sec + 5))
            if diff_sec > 0:
                time.sleep(diff_sec + 5)

        if r.status_code != 200:
            if r.status_code == 404:
                print("User has been deleted.")
                return 0
            elif r.status_code == 403:
                print("Forbidden :/")
            elif r.status_code == 401:
                print("User is been private.")
                return 0
            else:
                print("Error code: %d." %(r.status_code))
                time.sleep(50)
        else:
            goal_tweet_dict = json.loads(r.text)
            print("Got!")
            break

    json_dict = json.loads(r.text)
    collect_Whole_users.insert_one(json_dict)
    return json_dict["next_cursor"]

def get_ones(url, auth, user_id, collect_Whole_users):
    next_cursor = get_partial_followers_user_id(url = url, auth = auth, user_id = user_id, cur_cusor = -1, count = 200, collect_Whole_users = collect_Whole_users)
    safe_count = 0
    while next_cursor != 0:
        print(next_cursor)
        next_cursor = get_partial_followers_user_id(url = url, auth = auth, user_id = user_id, cur_cusor = next_cursor, count = 200, collect_Whole_users = collect_Whole_users)
        safe_count += 200
        if safe_count > follower_limitation:
            print("too much followers!")
            break

if __name__ == "__main__":
    client_loc = MongoClient("localhost", 27017)
    db_loc_Chat_Yojo_Bot = client_loc.Chat_Yojo_Bot
    collect_Whole_users = db_loc_Chat_Yojo_Bot.Whole_users

    url = "https://api.twitter.com/1.1/followers/list.json"
    AObj = OAuth()
    auth = OAuth1(AObj.api_key, AObj.api_secret, AObj.access_token, AObj.access_secret)

    trap_list_fname = "../../../Data/Chat-Yojo-Bot/trap.pickle"
    with open(trap_list_fname, "rb") as f:
        follower_list = pickle.load(f)

    pickle_fname = "../../../Data/Chat-Yojo-Bot/GetWholeFollowersIndex.pickle"
    if os.path.exists(pickle_fname):
        print("Yes!")
        with open(pickle_fname, "rb") as f:
            wholeIndex = pickle.load(f)
    else:
        wholeIndex = 0

    while wholeIndex < len(follower_list):
        print("###################")
        print("#   wholeIndex    #: ", wholeIndex)
        print("###################")

        user_id = follower_list[wholeIndex]
        get_ones(url, auth, user_id, collect_Whole_users)
        wholeIndex += 1
        with open(pickle_fname, 'wb') as f:
            pickle.dump(wholeIndex, f)
