# coding:utf-8
from __future__ import print_function
from selenium import webdriver
from selenium.common.exceptions import ElementNotVisibleException
from ast import literal_eval
from time import sleep
from bs4 import BeautifulSoup
# from html.parser import HTMLParser
import sys
import urllib.request
from http.client import BadStatusLine
from http.client import IncompleteRead
import codecs
import datetime
import pickle
import os.path
import pandas as pd
import numpy as np

def getNewNameSet():
    with urllib.request.urlopen('https://ask.fm/') as response:
        html = response.read()

    soup = BeautifulSoup(html.decode("utf-8"), "html.parser")
    askfmIdList = soup.body.footer.div.nav.get_text().split("\n")
    return set(askfmIdList[1:len(askfmIdList) - 1])

def DumpNewUtterReply(word):
    driver = webdriver.Chrome("C:\\Users\\David\\SPACE\\System\\chromedriver_win32\\chromedriver.exe")

    driver.get("https://ask.fm/" + word)

    while True:
        scroll_h = driver.execute_script("var h = window.pageYOffset; return h")
        judge = driver.execute_script("var m = window.pageYOffset; return m")
        previous_h = driver.execute_script("var h = window.pageYOffset; return h")
        #スクロール
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(3)
        after_h = driver.execute_script("var h = window.pageYOffset; return h")
        if previous_h == after_h:
            break
    print('load complete')

    page_source = driver.page_source

    questions = driver.find_elements_by_class_name("streamItem_header")
    answers = driver.find_elements_by_class_name("streamItem_content")

    # qas = [(q.find_element_by_tag_name('h2').text, a.text) for q, a in zip(questions, answers)]

    count = 0
    qas = []
    for q, a in zip(questions, answers):
        qatuple = (q.find_element_by_tag_name('h2').text, a.get_attribute('innerText'))
        qas.append(qatuple)
        count +=1
        if count == 10000:
            break

    print('find complete')

    Df = pd.DataFrame(np.empty([0, 0]))
    UtterList = []
    RepList = []

    for q, a in qas:
        if q == '' or a == '' or 'http' in q or 'http' in a:
            continue
        q = q.replace('\n', '')
        a = a.replace('\n', '')
        UtterList.append(q)
        RepList.append(a)
    Df["utter"] = UtterList
    Df["rep"] = RepList

    Df.to_json("../../../../Data/Chat-Yojo-Bot/AskFM/" + word + ".json")

    driver.quit()
    # webriver.Dispose()

def process():

    DoneNameSetFname =  "../../../../Data/Chat-Yojo-Bot/DoneNameSet.pickle"
    if os.path.exists(DoneNameSetFname):
        with open(DoneNameSetFname, "rb") as f:
            DoneNameSet = pickle.load(f)
    else:
        DoneNameSet = set([])

    NewNameSet = getNewNameSet()
    CheckedNameSet = NewNameSet.difference(DoneNameSet)

    print("DoneNameSet:\n", DoneNameSet)
    print("CheckedNameSet:\n", CheckedNameSet)

    for name in list(CheckedNameSet):
        print(name)
        DumpNewUtterReply(name)
        DoneNameSet.add(name)
        with open(DoneNameSetFname, "wb") as f:
            pickle.dump(DoneNameSet, f)
        sleep(5)

if __name__ == '__main__':
    while True:
        try:
            process()
            print("Waiting 10 minutes...")
            sleep(60 * 10)
        except:
            print("Some exception occured! :/")
            sleep(60 * 1)
