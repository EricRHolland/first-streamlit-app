#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing1

@author: hamzafarooq@ MABA CLASS
"""
import streamlit as st
import matplotlib
from shapely.geometry import Point
import matplotlib.pyplot as plt
import plotly.express as px

longitude_in = 0
latitude_in = 0
population_in = 0
median_income_in = 0


st.title("Eric Holland First Draft of App")
st.markdown("This is a demo ERIC app.")

st.markdown("This app takes the population, median income, longitude and latitude and outputs a predicted house value.")
st.markdown("The data is from a UCI repo on California housing prices that has way more than 4 variables.")
st.markdown("It uses random forest regression and grid search to fit then estimate inputs")
st.markdown("The output is the predicted home value.")
st.markdown("Im able to create it in Atom and Spyder but spent a lot of time getting it to work in streamlit.")
st.markdown("Spyder, Atom, Git Desktop are working fine, I wasnt able to let the user choose their preferred model among SVR, RF, Linear, etc.")

median_income_in1 = st.number_input("Enter your immediate area's median income in USD:", max_value = 500000, min_value = 0)
population_in1 = st.number_input("Enter your immediate area's population (max 6000):", max_value = 6000, min_value = 0)
longitude_in1 = st.number_input("Enter your district longitude betweeen -114 and -125:", min_value = -125)
latitude_in1 = st.number_input("Enter your district latitude between 32 and 42:",min_value = 32)
median_income_in = float(median_income_in1)/10000
population_in = int(population_in1)
longitude_in = int(longitude_in1)
latitude_in = int(latitude_in1)

@st.cache
def load_data(median_income_in, population_in, longitude_in, latitude_in):
    
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
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
    housevalues = housing[target_to_predict]
    
    scalerX = StandardScaler()
    housing_attributes = scalerX.fit_transform(housing_attributes)
    
    inputs = np.array([longitude_in,latitude_in, population_in,median_income_in])
    inputs = inputs.reshape(1,-1)
    inputs = scalerX.transform(inputs)
    
    dtr_clf = RandomForestRegressor()
    dtr_clf.fit(housing_attributes, housevalues.values.ravel())
    
    
    if median_income_in == 0 or population_in == 0 or longitude_in == 0 or latitude_in == 0:
        predicted_home_value = 0
    else: 
        predicted_home_value = dtr_clf.predict(inputs)
        predicted_home_value = float(predicted_home_value)
    return predicted_home_value



# print("How much would the average house cost in your district in 1990? \
#       Let's find out")


data = load_data(median_income_in, population_in, longitude_in, latitude_in)  
st.markdown("Your estimated California Home value is: ")
st.write(data)  

# lonlat = pd.DataFrame(longitude_in, latitude_in)
# lonlat
# fig = px.scatter_geo(geometry)
# fig.show()



#end

#python code that wont work in streamlit below:
# print("Enter your immediate area's median income in USD:")
# median_income_in = float(input())/10000
# print("Enter how many people there in your immediate area (max 6000):")
# population_in = int(input())
# print("Enter your district longitude:")
# longitude_in = float(input())
# print("Enter your district latitude:")
# latitude_in  = float(input())

# linreg = linear_model.LinearRegression()
# linreg = linreg.fit(housing_attributes,housevalues)
# linreg.score(housing_attributes,housevalues)
# linreg.coef_
# linreg.predict(inputs)


# Cut cross validation and grid search for computation time
# CROSS VALIDATION define inner and outer loops
# inner_cv = KFold(n_splits = 4, shuffle = True, random_state=50)
# outer_cv = KFold(n_splits = 4, shuffle = True, random_state=50)

# dtr = RandomForestRegressor()
# dtr_grid = {'max_depth': list(range(12,25))}

# dtr_clf = GridSearchCV(estimator=dtr, param_grid = dtr_grid, cv=inner_cv,
#                        return_train_score = False,
#                        refit=True)
# Run Grid search to find best estimators
# dtr_clf.best_params_

# import matplotlib.pyplot as plt
#     from sklearn import tree, neighbors, preprocessing, linear_model, svm, naive_bayes, linear_model
#     from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#     from sklearn.linear_model import LinearRegression
#     from sklearn.model_selection import cross_validate, train_test_split,cross_val_score
#     from sklearn.svm import SVR
#     from sklearn.tree import DecisionTreeRegressor
