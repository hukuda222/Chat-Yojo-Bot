import os
import sys
import json
import time
import pickle
import requests
import datetime
from pymongo import MongoClient
from requests_oauthlib import OAuth1

sys.path.append("../../../../Data/Chat-Yojo-Bot/")

from Key_and_Token import OAuth

def now_unix_time():
    return time.mktime(datetime.datetime.now().timetuple())

def check_already_exist(data, collect_Done):
    tweet_exist_dict = collect_Done.find_one({"id_str": data["in_reply_to_status_id_str"]})
    if tweet_exist_dict is not None:
        return True
    else:
        return False

# Functkon that access Twitter information dictionary, which returns it self if dictionary has "in_reply_to_status_id_str" key,
# else add key "in_reply_to_status_id_str" with value "null" then return it.

# For respective tweet seach using rest api, 900 times/15 min.
def search_tweet(data, collect_Conversation, collect_Base, collect_Done, auth, goal_id_str):
    # Lookup whether destination tweet exist in mongoDB.
    # If True insert current utter and response to the DB and recursively call search_tweet for detected tweet,
    # else if False using search api to get the tweet, then do same as True part.
    goal_tweet_dict = collect_Base.find_one({"id_str": goal_id_str})

    if goal_tweet_dict is not None:
        utter_rep = {"utter": goal_tweet_dict["text"], "rep": data["text"]}
        print("Existed tweet hit!")

    else:
        url = "https://api.twitter.com/1.1/statuses/show.json"
        while True:
            r = requests.get(url, auth = auth, params = {"id": int(goal_id_str)})

            limit = r.headers['x-rate-limit-remaining'] if 'x-rate-limit-remaining' in r.headers else 0
            print(limit)
            reset = float(r.headers['x-rate-limit-reset']) if 'x-rate-limit-reset' in r.headers else 0

            # Wait until limitation released
            if int(limit) == 0:
                diff_sec = reset - now_unix_time()
                print("sleep %d sec." % (diff_sec + 5))
                if diff_sec > 0:
                    time.sleep(diff_sec + 5)

            if r.status_code != 200:
                if r.status_code == 404:
                    print("Tweet has been deleted.")
                    return "Not found"

                elif r.status_code == 403:
                    print("Tweet is been locked.")
                    return "Locked"
                else:
                    print("Error code: %d." %(r.status_code))
                    time.sleep(5)
            else:
                goal_tweet_dict = json.loads(r.text)
                utter_rep = {"utter": goal_tweet_dict["text"], "rep": data["text"]}
                print("Got tweet!")
                break

    while True:
        try:
            collect_Done.insert_one(goal_tweet_dict) # Insert searched tweets.
            break;
        except:
            print("Insert to Done failed.")
            time.sleep(3)

    print("utter: ", utter_rep["utter"].encode("cp932", "replace").decode("cp932"))
    print("rep: ", utter_rep["rep"].encode("cp932", "replace").decode("cp932"))

    while True:
        try:
            collect_Conversation.insert_one(utter_rep) # Insert convsersation.
            break;
        except:
            print("Insert to Conversation failed.")
            time.sleep(3)

    # Check whether searched tweet is reply or not.
    # If is reply, then check whether that tweet exists in Done collection(exists in Done collection means that utter_rep already exist).
    goal_id_str_next = goal_tweet_dict.get("in_reply_to_status_id_str")

    if goal_id_str is not None and not check_already_exist(data, collect_Done):
        print(goal_id_str, ": goal_id_str")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        search_tweet(goal_tweet_dict, collect_Conversation, collect_Base, collect_Done, auth, goal_id_str_next)
    else:
        return "ReplyChainStops"

if __name__ == '__main__':
    print("#######################")
    print("# Start reply scraing #")
    print("#######################")
    print("\n")

    client_loc = MongoClient("localhost", 27017)
    db_loc_Chat_Yojo_Bot = client_loc.Chat_Yojo_Bot
    # Collect_Sample = db_loc_Chat_Yojo_Bot.Sample
    collect_Base = db_loc_Chat_Yojo_Bot.Base # Document's form is json which taken by twitter REST/STREAM api.
    collect_Done = db_loc_Chat_Yojo_Bot.Done # Document's form is json which taken by twitter REST/STREAM api.
    collect_Conversation = db_loc_Chat_Yojo_Bot.Conversation # document's form is: {"utter": "hoge", "resp": "fuga"}.

    AObj = OAuth()
    auth = OAuth1(AObj.api_key, AObj.api_secret, AObj.access_token, AObj.access_secret)

    cur_cursor = collect_Base.find({})
    cur_count = collect_Base.find({}).count()

    # pickle_fname = "Data/current_index.pickle"
    pickle_fname = "../../../../Data/Chat-Yojo-Bot/ReplyScraperIndex.pickle"

    if os.path.exists(pickle_fname):
        with open(pickle_fname, 'rb') as f:
            wholeIndex = pickle.load(f)
    else:
        wholeIndex = 0

    print("------------------------------------------------")
    while True:
        while wholeIndex < cur_count:
            print(wholeIndex)
            try:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                data = cur_cursor[wholeIndex]
                goal_id_str = data.get("in_reply_to_status_id_str")
                if goal_id_str is not None and not check_already_exist(data, collect_Done):
                    print(goal_id_str, ": goal_id_str")
                    # search_tweet(data, collect_Conversation, collect_Sample, collect_Done, auth, goal_id_str)
                    search_tweet(data, collect_Conversation, collect_Base, collect_Done, auth, goal_id_str)
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                else:
                    pass
            except TimeoutError:
                print("TimeoutError :/")
                time.sleep(5)
                continue

            wholeIndex += 1
            with open(pickle_fname, 'wb') as f:
                pickle.dump(wholeIndex, f)

        # See whether collection has new document.
        next_cursor = collect_Base.find({})
        next_count = collect_Base.find({}).count()

        # Sleep until new tweet got by WholeTimeLineScrape.
        while next_count <= cur_count:
            print("--------------------------------------------------------------------------------------------------------")
            print("No new tweet inserted, will sleep 10 sec waiting scraper get new tweet, current tweet quantity is {}.".format(next_count))
            print("--------------------------------------------------------------------------------------------------------")
            next_cursor = collect_Base.find({})
            next_count = collect_Base.find({}).count()
            time.sleep(10)

        cur_cursor = next_cursor
        cur_count = next_count
