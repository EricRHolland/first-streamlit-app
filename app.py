#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing1

@author: hamzafarooq@ MABA CLASS
"""

import streamlit as st
import pandas as pd

import plotly.express as px

st.title("Eric Holland First Draft of App")
st.markdown("This is a demo ERIC app.")

st.markdown("This app takes the population, median income, longitude and latitude and outputs a predicted house value.")
st.markdown("The data is from a UCI repo on California housing prices that has way more than 4 variables.")
st.markdown("For clarity's sake, I've stuck with 4 that a theoretical user can get off the internet")


import warnings
warnings.filterwarnings('ignore')
longitude_in = -122
latitude_in = 35
population_in = 3000
median_income_in = 4.5
print("Enter your immediate area's median income in USD:")
median_income_in = float(input())/10000
print("Enter how many people there in your immediate area (max 6000):")
population_in = int(input())
print("Enter your district longitude:")
longitude_in = float(input())
print("Enter your district latitude:")
latitude_in  = float(input())

print("How much would the average house cost in your district in 1990? \
      Let's find out")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, neighbors, preprocessing, linear_model, svm, naive_bayes, linear_model
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split,cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
housing = pd.read_csv(
    "https://raw.githubusercontent.com/EricRHolland/first-streamlit-app/main/housing.csv")
housing = pd.DataFrame(housing)




important_columns = ["longitude", "latitude","population","median_income"]
target_to_predict = ["median_house_value"]
newhouseframe = ["longitude", "latitude","population","median_income","median_house_value"]
housing = housing.dropna()

housing = housing[newhouseframe]
housing_attributes = housing[important_columns]
print(housing_attributes)
housevalues = housing[target_to_predict]
print(housevalues)

scalerX = preprocessing.StandardScaler()
housing_attributes = scalerX.fit_transform(housing_attributes)
print(housing_attributes)

inputs = np.array([longitude_in,latitude_in, population_in,median_income_in])
print(inputs)
inputs = inputs.reshape(1,-1)
print(inputs)
inputs = scalerX.transform(inputs)
print(inputs)


linreg = linear_model.LinearRegression()
linreg = linreg.fit(housing_attributes,housevalues)
linreg.score(housing_attributes,housevalues)
linreg.coef_
linreg.predict(inputs)



# CROSS VALIDATION define inner and outer loops
inner_cv = KFold(n_splits = 4, shuffle = True, random_state=50)
outer_cv = KFold(n_splits = 4, shuffle = True, random_state=50)

dtr = RandomForestRegressor()
dtr_grid = {'max_depth': list(range(12,25))}

dtr_clf = GridSearchCV(estimator=dtr, param_grid = dtr_grid, cv=inner_cv,
                       return_train_score = False,
                       refit=True)

# Run Grid search to find best estimators
dtr_clf.fit(housing_attributes, housevalues.values.ravel())
dtr_clf.best_params_

predicted_home_value = str(round(dtr_clf.predict(inputs),2))

print("The expected home value is: $",predicted_home_value,".")
