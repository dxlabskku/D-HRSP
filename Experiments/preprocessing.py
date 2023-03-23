import pandas as pd 
import numpy as np
import re
from konlpy.tag import Mecab
from khaiii import KhaiiiApi
import time

def preprocess(cd):
    data = pd.read_csv(f'{cd}')
    data['cleaned'] = [re.sub('[^A-Za-z0-9ㄱ-힇]', ' ', s) for s in data['content']]
    data['stripped'] = data['cleaned'].str.replace(" ", "")
    data = data[data['stripped'] != ''].reset_index(drop = True)
    
    mecab = Mecab()
    data['pos_tagged'] = data.cleaned.apply(mecab.pos)

    sens = []
    for i in range(len(data)):
        sen = ''
        sentence = data['pos_tagged'][i]
        for word, pos in sentence:
            if pos not in ('JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EF', 'EC' ):
                sen = sen + word + ' '
        sens.append(sen)

    data['clean_words_mecab'] = sens

    api = KhaiiiApi()
    sens = []
    for i in range(len(data)):
        sen = ''
        sentence = api.analyze(data['cleaned'][i])
        for word in sentence:
            for morph in word.morphs:
                if morph.tag not in ('JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EF', 'EC', 'MM', 'ZZ'):
                    sen = sen + morph.lex + ' '
        sens.append(sen)

    data['clean_words_khaiii'] = sens

    return data
