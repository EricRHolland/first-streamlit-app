#Class 2
#Feature engineering goes very far. Model will perform better with better data.
# good data with good feature engineering will win you competitions. 
# xg boost and lightgbm are the best for winning competitions




# Walmart 1.3 billion datapoints generated every day for Sales predictions. 
#label encoding is one of the ways to join multiple items together into one model
# BUT it calculates an output for each item separately. 
#following along with teh code given by the professor here. 
import mlflow
from google.colab import drive
drive.mount('/content/gdrive')
# in the 5 separate models, group items together, each item run separately
#slow moving items, fast moving items, how would you separate similar items ?
# stores, category?, whats the reasoning?
