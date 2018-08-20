import pandas as pd
import zenhan
import MeCab

mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')


df = pd.read_csv('./tweets.csv')
text_dict = {'<bos>': 0, '<eos>': 1}
text_list = list()
text_ids = list()
for text in df['text']:
    if text[0:2] != "RT" and text[0] != "@" and text[0:4] != "【定期】" and text[0:7] != "だれでも簡単！" and text[0:5] != "【ＤＬ中】":
        text = text.strip()
        text_list.append(text)
        text_ids.append(0)
        for n in mecab.parse(text).split("\n")[:-2]:
            word = n.split("\t")[0]
            if word not in text_dict:
                text_dict[word] = len(text_dict)
            text_ids.append(text_dict[word])
        text_ids.append(1)

"""
list_df = pd.DataFrame({'text': text_list})
list_df.to_csv('texts.csv')

list_df = pd.DataFrame({'text_ids': text_ids})
list_df.to_csv('texts_ids.csv')
"""
word_df = pd.DataFrame(
    {'word': [w for w, _ in sorted(text_dict.items(), key=lambda x: x[1])]})
word_df.to_csv('words.csv')
