import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing  import MinMaxScaler
import k_means_library as kml
import tqdm 

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# Read in dataset - need to change this to a URL endpoint rather than user desktop
df = pd.read_csv("C:/Users/kdb/Desktop/5800_final_project-main/bike-sharing.csv", index_col=0)

# check for null data
# this data set has been partially cleaned there are no null values 
#df.isnull().sum()

# a few of the variables have already been scaled. need to scale the target feature
df.describe()
####################################################################################################


# the hypothesis is that weather is a driving factor in the number of rentals
# plot the target variable against temp 
# plt.scatter(df['temp'],df['cnt'])
# plt.show()

# # There are no obvious clusters with this plotting. scaling the cnt data may help. 
scaler = MinMaxScaler()
df[["Scaled_count"]] = scaler.fit_transform(df[["cnt"]])


# plt.scatter(df['temp'],df['Scaled_count'])
# plt.show()
#######################################################################################################
#######################################################################################################
############# Grouping function to simplify graphing
# group by day -  avg the weathersit - avg temp - sum - cnt
# rescale cnt of this smaller dataset
df_grouped = df.copy(deep = True)

df_grouped = df_grouped[['dteday', 'weathersit','temp','cnt']]
df_grouped = df_grouped.groupby('dteday').agg(avg_wthr_sit = ('weathersit', 'mean'),avg_temp = ('temp', 'mean'), total_daily_count= ('cnt','sum'))
scaler_2 = MinMaxScaler()
df_grouped[["Scaled_count"]] = scaler_2.fit_transform(df_grouped[["total_daily_count"]])


df_grouped.head()
df_grouped.describe()


plt.scatter(df_grouped['avg_temp'],df_grouped['Scaled_count'])
plt.show()

plt.scatter(df_grouped['avg_wthr_sit'],df_grouped['Scaled_count'])
plt.show()
#########################################################################
# plot data  based on high medium low buckets using quantile cut function
# change the integer to any number you like to create that many cuts
df["temp_quantile_rank"] = pd.qcut(df['temp'], 3, labels=False)

df.describe()

# change the df.groupby('xxxx') to any of the discrete variables 
# seasons weekday and whatever bucket you use for temp are the most interesting
groups = df.groupby('weathersit')
for name, group in groups:
    plt.plot(group.temp, group.Scaled_count, marker='o', linestyle='', markersize=4, label=name)

plt.legend()
plt.show()
#######################################################################################################

# Use the elbow method to find the best number for k

all_data = df_grouped[['avg_wthr_sit', 'avg_temp','Scaled_count']].to_numpy(dtype=object)

avg_wthr_vs_scaled = df_grouped[['avg_wthr_sit','Scaled_count']].to_numpy()

avg_temp_vs_scaled = df_grouped[['avg_temp','Scaled_count']].to_numpy()

kml.elbow_method(all_data)
kml.k_means(avg_temp_vs_scaled,3, title = 'avg temp vs rider count', disp=True)
#kml.k_means(avg_wthr_vs_scaled,3,'weather situation vs rider count', disp=True)
#kml.k_means(all_data,3, 'All Data', disp=True)



























# k_range = range(1,10)
# SSE = {}

# for k in k_range:
    # kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df[['temp', 'Scaled_count']])
    # #df["clusters"] = kmeans.labels_ # only needed if we are going to do more work with the clusters but that is not a part of this analysis
    # SSE[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

# plt.figure()
# plt.plot(list(SSE.keys()), list(SSE.values()))
# plt.xlabel("Value of K")
# plt.ylabel("SSE")
# plt.title("Elbow Method")
# plt.show()
# #######################################################################################################
# # Convert data to numpy array

# data = df.to_numpy()

# print(data)
# print(data[:,[10,15]])

# x = data[:, 10]
# y = data[:, 15]

# plt.scatter(x,y)
# plt.show()

