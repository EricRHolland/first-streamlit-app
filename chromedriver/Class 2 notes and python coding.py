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



from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
print(data.head)
data = data[store_id == 'CA_1']
scaler = StandardScaler()
features = list('sell_price','rolling_mean_t30')
things_to_cluster_on = data[features]
scaled_features = scaler.fit_transform(features)
high volume high price low volumne low price how to encode within variables?
kmean = KMeans(n_clusters = 4)
kmean.fit(things_to_cluster_on)
y_kmeans = kmean.predict(X)

# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

import kneed
kl = KneeLocator(range(1,11), sse, curve = "convex"
)
new_k = kl.elbow
kmean = KMeans(n_clusters = new_k)
kmean.fit(things_to_cluster_on)
