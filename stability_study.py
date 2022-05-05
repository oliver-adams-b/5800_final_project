import numpy as np
import matplotlib.pyplot as plt
import tqdm
import k_means_l as kml

n = 100
bbox_performance = []
subsamp_performance = []

for j in tqdm.trange(n):
    k = np.random.randint(2, 7)
    d = np.random.randint(2, 10)
    
    #generate k test clusters in d-dimensional space.
    data = kml.generate_data(k,d,200)

    bbox_centroids, bbox_bins = kml.k_means(data, k, 
                                        init_function =  kml.generate_initial_centroids_from_bbox)
    subsamp_centroids, norm_bins = kml.k_means(data, k, 
                                        init_function =  kml.generate_initial_centroids_from_data)
    
    bbox_performance.append(kml.SSE(data, bbox_centroids))
    subsamp_performance.append(kml.SSE(data, subsamp_centroids))
    
plt.hist(bbox_performance, 
         color = "r", 
         label = "bbox method", 
         density= True, 
         bins = int(n/10)+1, 
         alpha = 0.5)

plt.hist(subsamp_performance, 
         color = "b", 
         label = "subsampling method", 
         density= True, 
         bins = int(n/10)+1, 
         alpha = 0.5)

plt.xlabel("Root Mean Squared Error, Goodness of Fit")
plt.ylabel("Normed Frequency of Occurrence")
plt.title("A Comparison of Initialization Techniques for K-Means")

plt.axvline(np.mean(bbox_performance), 
            0, 1, color = "r", linestyle = "dashed", 
            label = "subsampling mean")

plt.axvline(np.mean(subsamp_performance), 
            0, 1, color = "b", linestyle = "dashed", 
            label = "bbox mean")

plt.legend(loc = "best")     
