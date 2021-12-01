#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Eric Holland
Office hours 6:15pm Tuesday 11/30
Try to use pkl.load, otherwise get a working version and comment out

"""
import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
nlp = spacy.load("en_core_web_sm")
from spacy import displacy

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'

import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
import os

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv')

df['hotelName'].value_counts()
df['hotelName'].drop_duplicates()
df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='all_review')

# re combines and puts everything in lower case
import re
df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))
df = df_combined
df_sentences = df_combined.set_index("all_review")
df_sentences = df_sentences["hotelName"].to_dict()
df_sentences_list = list(df_sentences.keys())
len(df_sentences_list)
list(df_sentences.keys())[:5]

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

# gives an encoding of the full concatenated list of corpus
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
paraphrases = util.paraphrase_mining(model, corpus)

queries = ['Hotel close to Opera House',
          'Hotel with breakfast'
          ]

# Query sentences:
queries = []
inputquestion =input('What kind of hotel are you looking for?') 
queries.append(inputquestion)
query_embeddings = embedder.encode(queries,show_progress_bar=True)

from sentence_transformers import SentenceTransformer, util
import torch
embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
# Word cloud generates correctly
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print("(Score: {:.4f})".format(score))
        row_dict = df.loc[df['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['hotelName'] , "\n")
        wordcloud = WordCloud(width= 3000, height = 2000, random_state=30, background_color='white', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(str(corpus[idx]))
        plot_cloud(wordcloud)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        print()
