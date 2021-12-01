# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:40:14 2021

@author: EricH
"""

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

df = pd.read_csv("C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv")

## Use this to import spacy directly if using the Brain colab runtime or a custom colab runtime that includes spacy in its build.
import os
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy import displacy

# Unnamed: 0	review_body	review_date	hotelName	hotelUrl

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

print(df['hotelName'].value_counts())

df['hotelName'].drop_duplicates()

df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review.apply(''.join).reset_index(name='all_review')
  
df_combined.head().T

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

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Corpus with example sentences
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)
corpus_embeddings[0]

model = SentenceTransformer('all-MiniLM-L6-v2')
paraphrases = util.paraphrase_mining(model, corpus)
#query_embeddings_p =  util.paraphrase_mining(model, queries,show_progress_bar=True)

# import pickle as pkl
#upload a csv file, convereted that csv file after cleaning and converted to embedding
# if we dump the corpus into a pickle file, and then load the file and it will be saved in the system. 
# we havec saved the model, so we dont ahve to ever run the corpus 
# pkl.load()
# with open("/content/drive/MyDrive/BertSentenceSimilarity/Pickles/corpus_embeddings.pkl" , "wb") as file_:
#  pkl.dump(corpus_embeddings,file_)

queries = ['Hotel close to catacombs and has excellent breakfast',
           'Hotel which is close to the Louvre and is very high end with great shopping nearby'
           ]
query_embeddings = embedder.encode(queries,show_progress_bar=True)


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
print("\nTop 5 most similar sentences in corpus:")
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n=========================================================")
    print("==========================Query==============================")
    print("===",query,"=====")
    print("=========================================================")


    for idx, distance in results[0:closest_n]:
        print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
        print("Paragraph:   ", corpus[idx].strip(), "\n" )
        row_dict = df.loc[df['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['Hotel'] , "\n")
        # print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
        # print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")
        # print("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")
        print("-------------------------------------------")

from sentence_transformers import SentenceTransformer, util
import torch
embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


# Query sentences:
queries = ['Hotel at least 10 minutes away from Musee d Orsay',
           'walking distance to Gare du Nord'
           ]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
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
        print(corpus[idx], "(Score: {:.4f})".format(score))
        row_dict = df.loc[df['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['Hotel'] , "\n")
    # for idx, distance in results[0:closest_n]:
    #     print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
    #     print("Paragraph:   ", corpus[idx].strip(), "\n" )
    #     row_dict = df.loc[df['all_review']== corpus[idx]]
    #     print("paper_id:  " , row_dict['Hotel'] , "\n")
    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """

model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
embeddings = model.encode(corpus)
#print(embeddings)

# Query sentences:
queries = ['Hotel at least 10 minutes away from Musee d Orsay',
           'outside Gare du Nord'
           ]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print("(Score: {:.4f})".format(score))
        print(corpus[idx], "(Score: {:.4f})".format(score))
        row_dict = df.loc[df['all_review']== corpus[idx]]
        print("paper_id:  " , row_dict['Hotel'] , "\n")

hits = util.semantic_search(query_embedding, embeddings, top_k=5)
hits = hits[0]      #Get the hits for the first query
for hit in hits:
  print (hit)
  print("(Score: {:.4f})".format(hit['score']))
  print(corpus[hit['corpus_id']])
  row_dict = df.loc[df['all_review']== corpus[hit['corpus_id']]]
  print("paper_id:  " , row_dict['hotelName'] , "\n")

max_frequency=max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word]=word_frequencies[word]/max_frequency


sentence_tokens= [sent for sent in doc.sents]
print(sentence_tokens)

sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():                            
             sentence_scores[sent]=word_frequencies[word.text.lower()]
            else:
             sentence_scores[sent]+=word_frequencies[word.text.lower()]
             
from heapq import nlargest
select_length=int(len(sentence_tokens)*0.3)
select_length
summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
summary

final_summary=[word.text for word in summary]
final_summary
summary=''.join(final_summary)
summary

doc = df_combined['all_review'][0]

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer("summarize: " + doc, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(
    inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
)

print(tokenizer.decode(outputs[0]))

from transformers import pipeline

# Open and read the article
#f = open("article.txt", "r", encoding="utf8")
to_tokenize = doc

# Initialize the HuggingFace summarization pipeline
summarizer = pipeline("summarization",max_length = 500)
summarized = summarizer(to_tokenize, min_length=25, max_length=50)

# Print summarized text
print(summarized)

#embedder = SentenceTransformer('bert-base-nli-mean-tokens')

df = pd.read_csv("C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv")

dfnew = df['review_body'][0]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus = dfnew
corpus_embeddings = embedder.encode(corpus)