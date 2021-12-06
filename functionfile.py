# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:25:48 2021

@author: EricH
"""




# Import modules
import streamlit as st
from sklearn import tree, neighbors, preprocessing, linear_model, svm,naive_bayes
from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection
from sklearn.model_selection import cross_validate, train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report,make_scorer, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os

df = pd.read_csv('C:/Users/EricH/MachineLearning/try2/housing.csv')
housing = df
df.head()

important_columns = ["longitude", "latitude","population","median_income"]

X = df[important_columns]
X.head()
Y = df['median_house_value']
Y.head()

#scale the data for KNN Regression and others, dont have to scale Y
scalerX = StandardScaler()
X = scalerX.fit_transform(X)



#define the models
dtr = tree.DecisionTreeRegressor()
rfr = RandomForestRegressor()
knnr = KNeighborsRegressor()
svr = svm.SVR()

#define the grids and scoring

#decision tree regression instead of classifier

knnr_grid= {'n_neighbors':list(range(3,41)),
            'weights':["uniform", "distance"]}

svr_grid= [{'kernel': ['rbf'], 'gamma': [.0001,.001,.01,.1],
'C': [.1,1,10,100,1000]},
{'kernel': ['linear'], 'C': [.1,1,10,100,1000]}]

rfr_grid= {'max_depth': list(range(4,31))}


# CROSS VALIDATION define inner and outer loops
inner_cv = KFold(n_splits = 4, shuffle = True, random_state=50)
outer_cv = KFold(n_splits = 4, shuffle = True, random_state=50)


#Use grid search to train the models well.
rfr_clf5 = GridSearchCV(estimator=rfr, param_grid = rfr_grid, cv=inner_cv)
knnr_clf5 = GridSearchCV(estimator=knnr, param_grid = knnr_grid, cv=inner_cv)
svr_clf5 = GridSearchCV(estimator=svr, param_grid = svr_grid, cv=inner_cv)

rfr_score5 = cross_val_score(rfr_clf5, X=X, y=Y, cv=outer_cv)
knnr_score5 = cross_val_score(knnr_clf5, X=X, y=Y, cv=outer_cv)
svr_score5 = cross_val_score(svr_clf5, X=X, y=Y, cv=outer_cv)


print(rfr_score5.mean())
print(knnr_score5.mean())
print(svr_score5.mean())


rfr_clf5_fitted = rfr_clf5.fit(X,Y)
knnr_clf5_fitted = knnr_clf5.fit(X,Y)
svr_clf5_fitted = svr_clf5.fit(X,Y)



def rfr_fit(input):
    return print(rfr_clf5_fitted.predict(input))

def knnr_fit(input):
    return print(knnr_clf5_fitted.predict(input))

def svr_fit(input):
    return print(svr_clf5_fitted.predict(input))





#plot saves for calling from file or from here
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("firstcaligraph")


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

def tryingtofunctioncall():
    st.write("Hello people of the island")
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
