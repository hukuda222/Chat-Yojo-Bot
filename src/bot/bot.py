# coding: UTF-8
import datetime
import json
import os
import pickle
import re
import sys
import time
from threading import Timer

import requests
import zenhan as zh
from requests_oauthlib import OAuth1
import random

sys.path.append(os.getcwd().replace("/bot", ""))

from model import conversation_new as conv
from monologue import gentxt as gt

ng_words = ["殺", "死","しね","ころす","ころし"]


class gv:
    def __init__(self):
        self.tweet_limit = 49
        self.tweet_reset = None
        self.now_task = list()
        self.last_mention = "984355906388283392"


class RepeatedTimer(Timer):
    def __init__(self, interval, function, args=[], kwargs={}):
        Timer.__init__(self, interval, self.run, args, kwargs)
        self.thread = None
        self.function = function

    def run(self):
        self.thread = Timer(self.interval, self.run)
        self.thread.start()
        self.function(*self.args, **self.kwargs)

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel()
            self.thread.join()
            del self.thread


# tweetに関しては、上限に達すると403エラーが帰ってくるらしいので関係なかった
def get_limit(r):
    limit = int(r.headers['x-rate-limit-remaining']
                ) if 'x-rate-limit-remaining' in r.headers else None
    reset = float(r.headers['x-rate-limit-reset']
                  ) if 'x-rate-limit-reset' in r.headers else None
    return limit, reset


def get_time():
    return time.mktime(datetime.datetime.now().timetuple())


def reset_reset(g):
    g.tweet_reset = get_time() + float(60 * 30)
    g.tweet_limit = 49


def follow(auth, id):
    url = "https://api.twitter.com/1.1/friendships/create.json"
    requests.post(url, auth=auth, params={'user_id': str(id)}, timeout=100)


def follow_back(auth):
    url = "https://api.twitter.com/1.1/followers/ids.json"
    url2 = "https://api.twitter.com/1.1/friends/ids.json"
    r = requests.get(url, auth=auth, params={
                     'screen_name': 'trapyojo'}, timeout=100)
    r2 = requests.get(url2, auth=auth, params={
                      'screen_name': 'trapyojo'}, timeout=100)
    follower_user = list()
    follow_user = list()
    for u in r.iter_lines():
        follower_user = json.loads(u.decode("utf-8"))["ids"]
    for u in r2.iter_lines():
        follow_user = json.loads(u.decode("utf-8"))["ids"]
    for i in follower_user:
        if i not in follow_user:
            follow(auth, i)


def tweet(auth, params):
    url = 'https://api.twitter.com/1.1/statuses/update.json'
    requests.post(url, auth=auth, params=params, timeout=100)


def tweet_from_task(auth, g):
    if len(g.now_task) > 0 and g.tweet_limit > 0:
        task = g.now_task.pop()
        tweet_later = \
            Timer((g.tweet_reset - get_time()) / (1.01 + g.tweet_limit),
                  tweet, args=[auth, task])
        tweet_later.start()
        g.tweet_limit -= 1


def put_tweet(auth, text, tweet_id, g):
    if g.tweet_limit > 0:
        tweet_later = \
            Timer((g.tweet_reset - get_time()) / (1.01 + g.tweet_limit),
                  tweet, args=[auth, {'status': text,
                                      'in_reply_to_status_id': tweet_id}])
        tweet_later.start()
        g.tweet_limit -= 1
    else:
        g.now_task.append({'status': text, 'in_reply_to_status_id': tweet_id})


def init_last(auth, g):
    url = "https://api.twitter.com/1.1/statuses/mentions_timeline.json"

    tweets = requests.get(url, auth=auth, params={
                          "count": "200", "since_id": g.last_mention}).json()
    if len(tweets) > 0:
        g.last_mention = tweets[0]['id_str']


def get_unknown_word():
    if(random.random() > 0.3):
        return " ふぇぇ、わからないよぅ><"
    elif(random.random() > 0.3):
        return " わかんないや、ごめんね"
    else:
        return " わかんないから別の言葉で言って欲しいな"


def get_ng_word():
    if(random.random() > 0.3):
        return " お兄ちゃんに言っちゃダメって言われた事、言おうとしちゃった><"
    elif(random.random() > 0.3):
        return " 言っちゃダメな言葉は言わせないでね"
    else:
        return " またtwitterJPさんに怒られるから、もう過激なことは言わないよ"


def get_tweet(auth, g):
    url = "https://api.twitter.com/1.1/statuses/mentions_timeline.json"

    tweets = requests.get(url, auth=auth, params={
                          "count": "200", "since_id": g.last_mention}).json()
    if len(tweets) > 0:
        g.last_mention = tweets[0]['id_str']
        for tweet in tweets:
            try:
                got_tweet = tweet['text']
                patternScreenName = r"@[a-zA-Z0-9/:%#\$&\?\(\)~\.=\+\-_]+"
                patternUrl = r"https?://[a-zA-Z0-9/:%#\$&\?\(\)~\.=\+\-_]+"
                got_tweet = re.sub(patternScreenName, "", got_tweet)
                got_tweet = re.sub(patternUrl, "", got_tweet)
                got_tweet = re.sub(r'[\r|\t]', '', got_tweet)
                got_tweet = got_tweet.replace('\n', '')
                print('kitayo:' + got_tweet)
                utterLine = conv.parser(zh.z2h(got_tweet).lower())
                utterLineR = utterLine[::-1]
                text = "@" + str(tweet['user']['screen_name']) + " " + conv.conversation(utterLineR, conv.model,conv.dictionary, conv.id2wd)
                if any([text.find(ng) != -1 for ng in ng_words]):
                    text = "@" + str(tweet['user']['screen_name']) + str(get_ng_word())
                put_tweet(auth, text, tweet['id'], g)
            except:
                print("wakarazu")
                text = "@" + str(tweet['user']['screen_name']) + str(get_unknown_word())
                put_tweet(auth, text, tweet['id'], g)


def put_monologue(auth, g):
    if g.tweet_limit > 0:
        tweet(auth, {'status': gt.get_text()})
        g.tweet_limit -= 1


if __name__ == '__main__':
    auth = OAuth1(os.environ["CK"], os.environ["CS"],
                  os.environ["AT"], os.environ["AS"])
    g = gv()
    reset_reset(g)
    init_last(auth, g)

    t1 = RepeatedTimer(600, follow_back, [auth])
    t1.start()
    t2 = RepeatedTimer(60 * 30, reset_reset, [g])
    t2.start()
    t3 = RepeatedTimer(60 * 30 / 50, tweet_from_task, [auth, g])
    t3.start()
    t4 = RepeatedTimer(15 * 60 / 70, get_tweet, [auth, g])
    t4.start()
    t5 = RepeatedTimer(1800, put_monologue, [auth, g])
    t5.start()
