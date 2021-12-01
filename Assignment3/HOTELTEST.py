# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:46:41 2021

@author: EricH
"""

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl


st.title("Eric Holland First Draft of App")
st.markdown("This app shows a summary of each hotel?")

text = st.text_input("Enter your requirements or suggestions on where you want to stay:")

# new preprocessing file will havec the first run, then after that this hotel file will reference the first run
# but wont actually store the data

df = pd.read_csv("C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv")

dfnew = df['review_body'][0]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus = dfnew
corpus_embeddings = embedder.encode(corpus)

# with open("corpus_embeddings.pkl" , "wb") as file_:
#   pkl.dump(corpus_embeddings,file_)

with open('corpus_embeddings.pkl', 'rb') as file:
    myvar = pkl.load(file)


st.markdown(myvar)
# import pickle as pkl
#upload a csv file, convereted that csv file after cleaning and converted to embedding
# if we dump the corpus into a pickle file, and then load the file and it will be saved in the system. 
# we havec saved the model, so we dont ahve to ever run the corpus 
# pkl.load()
# with open("/content/drive/MyDrive/BertSentenceSimilarity/Pickles/corpus_embeddings.pkl" , "wb") as file_:
#   pkl.dump(corpus_embeddings,file_)



## Use this to import spacy directly if using the Brain colab runtime or a custom colab runtime that includes spacy in its build.

# st.table(df.head())
st.table(df.head())
# df['Hotel'].value_counts()


import os
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy


text = """Looking for a hotel in New York near Times Square with free breakfast and cheaper than $100 for 2nd June which is really kids friendly and has a swimming pool and I want to stay there for 8 days"""
doc = nlp(text)
sentence_spans = list(doc.sents)
displacy.render(doc, jupyter = True, style="ent")

stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'


import pandas as pd

from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl

embedder = SentenceTransformer('all-MiniLM-L6-v2')
#embedder = SentenceTransformer('bert-base-nli-mean-tokens')


#first install the library that would help us use BERT in an easy to use interface
#https://github.com/UKPLab/sentence-transformers/tree/master/sentence_transformers


# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # grab summary from summary hamza code
# text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'

# # Create and generate a word cloud image:
# wordcloud = WordCloud().generate(text)

# # Display the generated image:
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# st.pyplot()

# # plt.title('Topic '+str(first_choice))
# # plt.axis('off')
# # plt.tight_layout()
# # plt.imshow(wordcloud1)
# # plt.subplot(1,3,1)

# col1, col2, col3 = st.beta_columns(3)
# with col1:
#     st.header("A cat")
#     st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)
# with col2:
#     st.header("A dog")
#     st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)
# with col3:
#     st.header("An owl")
#     st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
