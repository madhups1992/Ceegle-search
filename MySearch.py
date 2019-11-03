import datetime
import re
import numpy as np
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template
from math import sqrt
from sklearn import metrics
from sklearn.externals import joblib

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait
import re
import requests
import pandas as pd
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import random
from random import uniform
from datetime import datetime
import json

import os
import tweepy as tw
from datetime import date, timedelta
import nltk

nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

import gensim
from gensim.models import Word2Vec
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
import spacy
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns


from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import models

from nltk import FreqDist


# results[0].get_attribute("href")
def youtube_search(search_val):
    driver = webdriver.Chrome()
    driver.get("https://www.youtube.com/")
    # driver.get(baseUrl + "/results?search_query=selenium&sm=3")
    # driver.findElement(By.id("masthead-search-term"))
    search = driver.find_element_by_css_selector("input#search")

    # youtube Search
    search.send_keys(search_val)
    time.sleep(0.5)
    search.send_keys(Keys.RETURN)

    Search_page_url = driver.current_url
    time.sleep(2)
    Search_page_results = driver.find_elements_by_id("video-title")

    results_df = pd.DataFrame(columns=['href', 'title'])

    for result in Search_page_results:
        # attrs = driver.execute_script('var items = {}; for (index = 0; index < arguments[0].attributes.length; ++index) { items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; return items;', result[0])
        results_df = results_df.append({'href': result.get_attribute("href"), 'title': result.get_attribute('title')},
                                       ignore_index=True)

    driver.quit()
    return (results_df)


def results_search(posts):
    results = []
    for post in posts:
        a = re.search(r'\w*\nhttps', post.text)
        if (a):
            results.append(post.text)

    results_df = pd.DataFrame(columns=['href', 'title'])

    for result in results:
        b = result.split('\n')
        results_df = results_df.append({'href': b[1].replace(' › ', '/'), 'title': b[0]}, ignore_index=True)
    return (results_df)


def google_search(search_val, page):
    browser = webdriver.Chrome()
    browser.get('http://www.google.com')
    search = browser.find_element_by_name('q')

    # Google Search
    search.send_keys(search_val)
    time.sleep(2)
    search.send_keys(Keys.RETURN)  # hit return after you enter search text

    results_df = pd.DataFrame(columns=['href', 'title'])
    # For more pages
    a = 0;
    while a <= page - 1:
        time.sleep(2)
        posts = browser.find_elements_by_class_name("r")
        df = results_search(posts)
        results_df = results_df.append(df, ignore_index=True)
        time.sleep(1.5)
        browser.find_element_by_xpath("//*[contains(local-name(), 'span') and contains(text(), 'Next')]").click()
        a = a + 1

    browser.quit()

    return (results_df)


# Text data cleaning
def text_cleaning_part1(X):
    tweets = []
    stemmer = WordNetLemmatizer()

    for s in range(0, len(X)):
        tweet = re.sub(r'\W', ' ', str(X[s]))  # specal char

        tweet = tweet.lower()  # lowercase

        tweet = re.sub(r'\s+[A-z]\s+', ' ', tweet)  # single characters like 'a,i'

        tweet = re.sub(r'\^[A-z]\s+', ' ', tweet)  # single characters in begining

        tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)  # multiple spaces

        tweet = re.sub(r'^b\s+', ' ', tweet)  # Removing prefixed 'b'

        tweet = tweet.split()  # Lemmatization

        # tweet = [stemmer.lemmatize(word) for word in tweet]
        tweet = ' '.join(tweet)

        tweets.append(tweet)

    return (tweets)


def query_processor(url_query):
    driver = webdriver.Chrome()
    driver.get(url_query)

    body = driver.find_element_by_tag_name('body')

    for _ in range(60):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.7)

    timeline = driver.find_elements_by_class_name('TweetTextSize')
    results = []
    for i in timeline:
        results.append(i.text)

    driver.quit()

    return (results)


def scrape_tweets(searchword):
    # conspiracy url
    url_query_cons = 'https://twitter.com/search?q=%23conspiracy%20' + searchword + '&src=typed_query'
    cons = query_processor(url_query_cons)
    df_cons = pd.DataFrame({'tweets': cons})
    df_cons['conspiracy'] = True

    # non conspiracy url
    url_query_noncons = 'https://twitter.com/search?q=' + searchword + '&src=typed_query'
    noncons = query_processor(url_query_noncons)
    # noncons = text_cleaning(noncons)
    df_noncons = pd.DataFrame({'tweets': noncons})
    df_noncons['conspiracy'] = False

    # concatenating
    df = pd.concat([df_cons, df_noncons], ignore_index=True)

    return (df)


# Cleaning text
# text cleaning continution
nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, tags=['NOUN', 'ADJ', 'VERB', 'PROPN', 'PART']):  # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp("".join(sent))
        # print("1",doc)
        review = [token.lemma_ for token in doc if token.pos_ in tags]
        review = ' '.join(review)
        output.append(review)
    return output


def remove_stopwords(rev,stop_words):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


# Removing url from tweets
def removing_url(df_v1, col_name):
    for i in range(len(df_v1[col_name])):
        a = re.search(r'([A-z0-9]+\.[.A-z0-9/]+)', df_v1[col_name][i])
        if (a):
            string = df_v1[col_name][i]
            df_v1[col_name][i] = string.replace(a.group(1), '')
    return (df_v1)


# Removing null
def removing_null(df_v1, col_name):
    null_tweets = []
    for i in range(len(df_v1)):
        if (len(df_v1[col_name][i]) == 0):
            null_tweets.append(i)
    df_v1 = df_v1.drop(null_tweets).reset_index(drop=True)
    return (df_v1)


def cleaning_text(df_v1, col_name,stop_words):
    df_v1 = removing_url(df_v1, col_name)
    df_v1[col_name] = text_cleaning_part1(df_v1[col_name])
    df_v1[col_name] = [remove_stopwords(r.split(),stop_words) for r in df_v1[col_name]]
    df_v1[col_name] = lemmatization(df_v1[col_name])
    # df_v1 = removing_null(df_v1,col_name)
    return (df_v1)


# Unigram_converter
def bigram_converter(review):
    token_ = [doc.split(" ") for doc in review]
    bigram = Phrases(token_, min_count=1, threshold=2, delimiter=b' ')

    bigram_phraser = Phraser(bigram)

    bigram_token = []
    for sent in token_:
        bigram_token.append(bigram_phraser[sent])

    return (bigram_token)


def unigram_converter(tweets):
    token_ = [doc.split(" ") for doc in tweets]

    return (token_)


# frequent words finder


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    return (d)


def Tfidfvec(text_col,stop_words):
    # Building Tfidf vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_col)
    X.todense()

    # vectorizer.inverse_transform(X)
    col_names = vectorizer.get_feature_names()
    X_list = X.toarray().tolist()
    return (X_list, col_names)


def remove_nonfreq_words(rev, total_words):
    rev_new = " ".join([i for i in rev if i in total_words])
    return rev_new


import math
from math import sqrt


def cosine_similarity(vec1, vec2):
    v11 = 0
    v12 = 0
    v22 = 0

    for i in range(len(vec1)):
        v11 += (vec1[i] * vec1[i])
        v22 += (vec2[i] * vec2[i])
        v12 += (vec1[i] * vec2[i])

    cosine_sim = (v12 / math.sqrt(v11 * v22))

    return (cosine_sim)


def euclidean_distance(vec1, vec2):
    dist = 0
    for i in range(len(vec1) - 1):
        # (x1-y1)^2
        diff = vec1[i] - vec2[i]
        dist += pow((diff), 2)
    distance = sqrt(dist)
    return distance


# finding the most similar neighbors
def get_near_neibr(train, test_row, num_neibr):
    distances = []
    neighbors = []

    # Getting all the distances
    for row in train:
        dist = euclidean_distance(test_row, row[:-1])
        distances.append((row, dist))
    # soting them to take top num_neighbor
    distances = sorted(distances, key=lambda x: x[1])

    for i in range(num_neibr):
        neighbors.append(distances[i][0])

    return neighbors


def predict_classification(train, test_row, num_neibr):
    neighbors = get_near_neibr(train, test_row, num_neibr)

    train_output = [row[-1] for row in neighbors]

    unique, counts = np.unique(train_output, return_counts=True)
    output = np.asarray((unique, counts)).T
    prediction = max(set(train_output), key=train_output.count)

    return prediction


def KNN(train_set, test_set, num_neibr):
    predictions = []
    # finding the max vote of the nearest neighbour
    for row in test_set:
        output = predict_classification(train_set, row, num_neibr)
        predictions.append(output)

    return (predictions)


def google_prediction(X_train, google_df,stop_words,total_words):
    google_df1 = google_df.copy()
    google_clean_df = cleaning_text(google_df1, col_name='title',stop_words=stop_words)
    google_clean_df['title'] = [remove_nonfreq_words(r.split(), total_words) for r in google_clean_df['title']]
    X_list, col_names = Tfidfvec(text_col=google_clean_df['title'],stop_words=stop_words)
    google_v2 = pd.DataFrame(X_list, columns=col_names)
    google_v3 = google_v2.merge(google_clean_df, how='outer', left_index=True, right_index=True)

    X_google = google_v3.iloc[:, 0:len(google_v3.columns) - 2].values
    res = KNN(X_train, X_google, num_neibr=5)

    google_clean_df['pred'] = res
    for i in range(len(google_clean_df['title'])):
        isnotempty = google_clean_df['title'][i]
        if (isnotempty == ''):
            google_clean_df['pred'][i] = False

    return (google_clean_df)


def final_prediction(searchword):
    google_df = google_search(search_val=searchword, page=4)

    searchword_first = searchword.split()[0]
    df = scrape_tweets(searchword_first)

    stop_words = stopwords.words('english')
    stop_words = stop_words + ['conspiracy', 'theory', 'http', 'conspiracytheory']
    stop_words = ['conspiracy', 'theory', 'http', 'conspiracytheory', searchword]

    df_v1 = df.copy()
    # Cleaning text
    df_v1 = cleaning_text(df_v1, col_name='tweets',stop_words=stop_words)
    pos_words = freq_words(df_v1[df_v1['conspiracy'] == True]['tweets'], 40)
    neg_words = freq_words(df_v1[df_v1['conspiracy'] == False]['tweets'], 40)
    total_words = list(pos_words['word']) + list(neg_words['word'])

    df_v1['tweets'] = [remove_nonfreq_words(r.split(), total_words) for r in df_v1['tweets']]
    df_v1 = removing_null(df_v1, 'tweets')

    X_list, col_names = Tfidfvec(text_col=df_v1['tweets'],stop_words=stop_words)
    df_v2 = pd.DataFrame(X_list, columns=col_names)
    df_v3 = df_v2.merge(df_v1, how='outer', left_index=True, right_index=True)

    X = df_v3.iloc[:, df_v3.columns != 'tweets'].values
    y = df_v3.iloc[:, len(df_v3.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    google_clean_df = google_prediction(X_train, google_df,stop_words,total_words)
    google_df['pred'] = google_clean_df['pred']

    return (google_df)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('search_page.html')

@app.route('/search/<search_key>/')
def success(search_key):
    google_search_df = final_prediction(search_key)

    pred = list(google_search_df['pred'])
    ref = list(google_search_df['href'])
    title = list(google_search_df['title'])

    return render_template('result_page.html', search_key=search_key, ref=ref, title=title, pred=pred,
                           length=len(title))



@app.route('/search', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        search_key = request.form['search_key']
        print(search_key)
        return redirect(url_for('success', search_key = search_key))
    else:
        search_key = request.form['search_key']
        return redirect(url_for('success', search_key = search_key))

if __name__ == '__main__':
    app.run(debug=True)
