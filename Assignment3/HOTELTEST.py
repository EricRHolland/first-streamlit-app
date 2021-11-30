# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:46:41 2021

@author: EricH
"""

import streamlit as st
import pandas as pd


longitude_in = 0
latitude_in = 0
population_in = 0
median_income_in = 0


st.title("Eric Holland First Draft of App")
st.markdown("This app shows a summary of each hotel?")


data = pd.read_csv("C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv")


