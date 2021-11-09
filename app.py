#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: hamzafarooq@ MABA CLASS
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import os
import tarfile
import urllib.request
from __future__ import division, print_function, unicode_literals

np.random.seed(50)

# To plot pretty figures
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("MachineLearning", "try2", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()




st.title("Welcome to MABA Class")
st.markdown("This is a demo ERIC app.")
st.markdown("ERIC, hello world!..")
st.markdown("WHY ISNT THIS WORKING")


@st.cache(persist=True)
def load_data():
    df = pd.read_csv("https://datahub.io/machine-learning/iris/r/iris.csv")
    return(df)



def run():
    st.subheader("Iris Data Loaded into a Pandas Dataframe.")

    df = load_data()



    disp_head = st.sidebar.radio('Select DataFrame Display Option:',('Head', 'All'),index=0)



    #Multi-Select
    #sel_plot_cols = st.sidebar.multiselect("Select Columns For Scatter Plot",df.columns.to_list()[0:4],df.columns.to_list()[0:2])

    #Select Box
    #x_plot = st.sidebar.selectbox("Select X-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=0)
    #y_plot = st.sidebar.selectbox("Select Y-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=1)


    if disp_head=="Head":
        st.dataframe(df.head())
    else:
        st.dataframe(df)
    #st.table(df)
    #st.write(df)


    #Scatter Plot
    fig = px.scatter(df, x=df["sepallength"], y=df["sepalwidth"], color="class",
                 size='petallength', hover_data=['petalwidth'])

    fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    st.write("\n")
    st.subheader("Scatter Plot")
    st.plotly_chart(fig, use_container_width=True)


    #Add images
    #images = ["<image_url>"]
    #st.image(images, width=600,use_container_width=True, caption=["Iris Flower"])





if __name__ == '__main__':
    run()
