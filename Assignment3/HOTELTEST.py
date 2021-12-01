# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:46:41 2021

@author: EricH
"""

import streamlit as st
import pandas as pd




st.title("Eric Holland First Draft of App")
st.markdown("This app shows a summary of each hotel?")


data = pd.read_csv("C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv")


## Use this to import spacy directly if using the Brain colab runtime or a custom colab runtime that includes spacy in its build.
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


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# grab summary from summary hamza code
text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot()

# plt.title('Topic '+str(first_choice))
# plt.axis('off')
# plt.tight_layout()
# plt.imshow(wordcloud1)
# plt.subplot(1,3,1)

col1, col2, col3 = st.beta_columns(3)
with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)
with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", use_column_width=True)
with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)
