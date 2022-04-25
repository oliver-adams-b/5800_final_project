import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing  import MinMaxScaler

# Read in dataset - need to change this to a URL endpoint rather than user desktop
df = pd.read_csv("C:/Users/kdb/Desktop/Final_Algo_project/bike-sharing_1.csv", index_col=0)

# check for null data
# this data set has been partially cleaned there are no null values 
#df.isnull().sum()

# a few of the variables have already been scaled. need to scale the target feature
df.describe()
####################################################################################################


# the hypothesis is that weather is a driving factor in the number of rentals
# plot the target variable against temp 
plt.scatter(df['temp'],df['cnt'])
plt.show()

# There are no obvious clusters with this plotting. scaling the cnt data may help. 
scaler = MinMaxScaler()
df[["Scaled_count"]] = scaler.fit_transform(df[["cnt"]])

plt.scatter(df['temp'],df['Scaled_count'])
plt.show()
#######################################################################################################

# plot data  based on high medium low buckets using quantile cut function
# change the integer to any number you like to create that many cuts
df["temp_quantile_rank"] = pd.qcut(df['temp'], 3, labels=False)

df.describe()

# change the df.groupby('xxxx') to any of the discrete variables 
# seasons weekday and whatever bucket you use for temp are the most interesting
groups = df.groupby('season')
for name, group in groups:
    plt.plot(group.temp, group.Scaled_count, marker='o', linestyle='', markersize=4, label=name)

plt.legend()

#######################################################################################################

# Use the elbow method to find the best number for 4

k_range = range(1,10)
SSE = {}

for k in k_range:
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df[['temp', 'Scaled_count']])
    #df["clusters"] = kmeans.labels_ # only needed if we are going to do more work with the clusters but that is not a part of this analysis
    SSE[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

plt.figure()
plt.plot(list(SSE.keys()), list(SSE.values()))
plt.xlabel("Value of K")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.show()
#######################################################################################################


data = df.to_numpy()

print(data)
print(data[:,[10,15]])

x = data[:, 10]
y = data[:, 15]

plt.scatter(x,y)
plt.show()

