# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:48:08 2021

@author: EricH
"""


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




# #see if it works as expected
# strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# #removing income category attribute so data is back to its pre-operative state
# for set_ in (strat_train_set, strat_test_set):
#     set_.drop("income_cat", axis=1, inplace=True)

# # plot the data geographically
# housing.plot(kind="scatter", x="longitude", y="latitude")

# # try to use the datapoints as shading instead
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha = .1)
# #plot it together using population and lat
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.4,
#             s = housing["population"]/100, label= "population", figsize = (10,7),
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#             )
# plt.legend()

# thingstoscatter = ["median_house_value", "median_income", "total_rooms",
#                    "housing_median_age"]
# scatter_matrix(housing[thingstoscatter], figsize=(13,7))

# #zoom in on median income
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=.15)

# #clean up the data set for new operations
# housing = strat_train_set.drop("median_house_value", axis=1)
# housing_labels = strat_train_set["median_house_value"].copy()



# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScalar
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor



# imputer = SimpleImputer(strategy="median")

# housing_num = housing.drop("ocean_proximity", axis=1)

# imputer.fit(housing_num)
# X = imputer.fit(housing_num)

# housing_tr = pd.DataFrame(X, columns = housing_num.columns,
#                           index = housing_num.index)
# housing_cat = housing[["ocean_proximity"]]

# ordinal_encoder = OrdinalEncoder()
# # fit the imputer to the dataset
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# ordinal_encoder.categories_
# # one hot encoder allows you to get the missing values in a categorical variable

# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# housing_cat_1hot
# # convert the encoder to a sparse array in numpy
# housing_cat_1hot.toarray()

# cat_encoder.categories_

# # now we need to create a transformer.
# # needs to work with pipelines and duck typing.

# # need to create a class and implement three methods
# # fit returning self, transform(), fit_transform()

# #create small transformer class that adds previous combined attributes



# rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def _init_(self, add_bedrooms_per_room = True): # no *args or **kargs
#         self.add_bedrooms_per_room = add_bedrooms_per_room
#     def fit(self,X,y=None):
#         return self # nothing else to do
    
#     def transform(self,X):
#         rooms_per_household = X[:, rooms_ix] / X[:,households_ix]
#         population_per_household = X[:, population_ix] / X[:,households_ix]
#         if self.add_bedrooms_per_room: 
#             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
#             return np.c_[X, rooms_per_household, population_per_household,
#                          bedrooms_per_room]
#         else:
#             return np.c_[X, rooms_per_household, population_per_household]
# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
# housing_extra_attribs = attr_adder.transform(housing.values)


# #need feature scaling. algos dont perform well when numerical attributes have different scales

# #two ways to get attributes to have the same scale: min max scaling and standardization
# #you MUST fit the scalars to the training data, not to the full dataset.

# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy = "Median")),
#     ('attribs_adder', CombinedAttributesAdder()),
#     ('std_scaler', StandardScalar()),
#     ])

# housing_num_tr = num_pipeline.fit_transform(housing_num)

# # Another way to transform the data as long as its in pandas DF



# num_attribs = list(housing_num)
# cat_attribs = ["ocean_proximity"]

# full_pipeline = ColumnTransformer([
#     ("num", num_pipeline, num_attribs),
#     ("cat", OneHotEncoder(), cat_attribs),
#     ])
# #constructor requires a list of tuples
# housing_prepared = full_pipeline.fit_transform(housing)

# #finally ready to select and train a model

# # start with linear regression
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)



# some_data = housing.iloc[5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)

# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

# # check RMSE on the whole training set


# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_rmse
# # rmse is really big, almost half of the training data average. This is underfitting.

# #instead, try a decision tree regressor, can find nonlinear relationships in data.



# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)

# #Then evaluate on training set
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# tree_mse

# #overfit the training data! Now we have to redo some of our cross validation

# #use Scikit's K-fold cross validation, randomly splits the data and then uses
# # gives you 10 different scores depending on which fold was the best

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring = "neg_mean_squared_error", cv = 10)
# tree_rmse_scores = np.sqrt(-scores)


# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard Dev.:", scores.std())
    

# from sklearn.ensemble import RandomForestRegressor

# forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
# forest_reg.fit(housing_prepared, housing_labels)


# from sklearn.svm import SVR

# svm_reg = SVR(kernel="linear")
# svm_reg.fit(housing_prepared, housing_labels)
# housing_predictions = svm_reg.predict(housing_prepared)
# svm_mse = mean_squared_error(housing_labels, housing_predictions)
# svm_rmse = np.sqrt(svm_mse)
# svm_rmse

# from sklearn.model_selection import GridSearchCV

# param_grid = [
#     # try 12 (3×4) combinations of hyperparameters
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     # then try 6 (2×3) combinations with bootstrap set as False
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#   ]

# forest_reg = RandomForestRegressor(random_state=42)
# # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error', return_train_score=True)
# grid_search.fit(housing_prepared, housing_labels)

# grid_search.best_params_
# grid_search.best_estimator_

# feature_importances = grid_search.best_estimator_.feature_importances_
# feature_importances

# extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# #cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# sorted(zip(feature_importances, attributes), reverse=True)



# full_pipeline_with_predictor = Pipeline([
#         ("preparation", full_pipeline),
#         ("linear", LinearRegression())
#     ])

# full_pipeline_with_predictor.fit(housing, housing_labels)
# full_pipeline_with_predictor.predict(some_data)

# my_model = full_pipeline_with_predictor

