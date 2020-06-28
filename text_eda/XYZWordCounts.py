import sys
import pandas as pd
import csv
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pylab as plt
import math
from scipy import stats
import geopandas as gpd

from functools import reduce
import importlib
import mr_word_count
importlib.reload(mr_word_count)
from mr_word_count import MRWordFrequencyCount
from mrjob.job import MRJob
import mapreduce as mr
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from collections import Counter

import glob



def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words



def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(i) for i in text]
    return lem_text

if __name__=='__main__':
    #for x in NeighborhoodInputs df = pd.read_csv('Chelsea.csv')
    for inputCSV in glob.glob('*.csv'):
        df = pd.read_csv(inputCSV)
        df['description'] = df['description'].str[:161]
        tokenizer = RegexpTokenizer('\s+', gaps=True)
        df['description'] = df['description'].apply(lambda x: tokenizer.tokenize(x.lower()))
        df['description'] = df['description'].apply(lambda x: remove_stopwords(x))
        lemmatizer = WordNetLemmatizer()
        df['description'].apply(lambda x: word_lemmatizer(x))
        df['ZIPCODE'] = df['ZIPCODE'].astype(int)
        df['description'] = df['description'].astype(str)
        
        a = df['description'].str.cat()
        words_a = a.split()
        wordCount_a = Counter(words_a)
        
        df1 = pd.DataFrame.from_dict(wordCount_a, orient='index').reset_index()
        df1 = df1.rename(columns={"index": "word", 0:'count'})
        df1['word'] = df1['word'].str.replace(r'[^\w\s]+', '')
        df1 = df1[df1.word != '']
        df1 = df1.groupby(['word'], as_index=False)['count'].sum()

        if df1[df1['word'] == "york"].size>0:
            b = df1.loc[df1['word'] == 'new', 'count'].item()
            c = df1.loc[df1['word'] == 'york', 'count'].item()

            if b == c:
                df1.drop(df1[ df1['word'] == 'york' ].index , inplace=True)
                df1.drop(df1[ df1['word'] == 'new' ].index , inplace=True)
            else:
                d = b - c
                df1.loc[df1.word == "new", "count"] = d
                df1.drop(df1[ df1['word'] == 'york' ].index , inplace=True)
        
 
        df1 = df1.sort_values('count', ascending=False)
        df100 = df1.head(150)
        df100 = df100.reset_index(drop=True)
        #df100.to_csv('output.csv')
        df100.to_csv('../NeighborhoodOutputs/'+inputCSV + '_output.csv', index=False)
