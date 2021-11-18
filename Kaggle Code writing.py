# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:03:55 2021

@author: EricH
"""

# pandas
# .HEAD
# pandas dataframename.head(n=numeric digit to show how many rows you want)


# .LOC
# pandas .loc select only those rows who have specific attributes. 
# # Set value for an entire row
# df.loc['cobra'] = 10
# df
#             max_speed  shield
# cobra              10      10
# viper               4      50
# sidewinder          7      50
# # Get one store data
# def one_store(store):
#     '''Selecting only one store (Redefining input data to include only for one store)'''
#     calendar, sell_prices, sales_train_validation, submission = read_data()
#     store_sales_train_validation = sales_train_validation.loc[sales_train_validation['store_id'] == store]
#     store_sell_prices =  sell_prices.loc[sell_prices['store_id'] == store]
#     data = melt_and_merge(calendar, store_sell_prices, store_sales_train_validation, submission, nrows = 50000, merge = True)
#     return data
# .loc users the id as the thing to match to so it only selects those rows, pretty much operates as a filter.













def one_store(store):
    #'''Selecting only one store (Redefining input data to include only for one store)'''
    calendar, sell_prices, sales_train_validation, submission = read_data()
    store_sales_train_validation = sales_train_validation.loc[sales_train_validation['store_id'] == store]
    store_sell_prices =  sell_prices.loc[sell_prices['store_id'] == store]
    data = melt_and_merge(calendar, store_sell_prices, store_sales_train_validation, submission, nrows = 50000, merge = True)
    return data

# Transform, Train and Validate on entire one store data
transform_train_and_eval(one_store('CA_1'),'submission.csv')
print("\n")

# Reset back to original form and not include any transformation done above
data = one_store('CA_1')

# Getting the total demand for each individual dept
eda_data = data.groupby(['dept_id']).agg('sum')['demand'].reset_index()

# Extract the demand percentage 
eda_data['demand_percent'] = (eda_data['demand'] / eda_data['demand'].sum()) * 100

# Sort the data based on demand percentage
eda_data = eda_data.sort_values(by=['demand_percent'])

# Plot the dept and their respective demands
eda_data.plot.bar(x='dept_id',y='demand_percent', color=['r','r','r','orange','orange','orange','g'])

# Print Summary
print('\nBased on the above plot, it is evident that we can categorize into three different groups as below.')



