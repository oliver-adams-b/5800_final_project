import numpy as np
import matplotlib.pyplot as plt
import tqdm
import k_means_sandbox as kms

n = 400
bbox_performance = []
norm_performance = []

for j in tqdm.trange(n):
    
    #generate k test clusters in d-dimensional space:
    k = np.random.randint(2, 7)
    d = np.random.randint(2, 10)
    
    #randomly generate the locations of the random clusters, a set of k points on the plane
    means = np.random.randint(-100, 100, (k, d))

    #randombly generate the variance of the cluster surrounding each point
    stds = np.random.randint(1, 40, (k, d, d))

    #now generate the test data from the locations and variances:
    data = kms.generate_data(means,
                         stds, 
                         200)
    
    bbox_centroids, bbox_bins = kms.k_means(data, k, 
                                        init_function =  kms.generate_initial_centroids_from_bbox)
    norm_centroids, norm_bins = kms.k_means(data, k, 
                                        init_function =  kms.generate_initial_centroids_from_data)
    
    bbox_performance.append(kms.SSE(data, bbox_centroids))
    norm_performance.append(kms.SSE(data, norm_centroids))
    
plt.hist(bbox_performance, 
         color = "r", 
         label = "bbox method", 
         density= True, 
         bins = int(n/4), 
         alpha = 0.5)

plt.hist(norm_performance, 
         color = "b", 
         label = "subsampling method", 
         density= True, 
         bins = int(n/4), 
         alpha = 0.5)

plt.xlabel("Sum of Squared Error, Goodness of Fit")
plt.ylabel("Normed Frequency of Occurrence")
plt.title("A Comparison of Initialization Techniques for K-Means")

plt.axvline(np.mean(bbox_performance), 
            0, 1, color = "r", linestyle = "dashed", 
            label = "subsampling mean")

plt.axvline(np.mean(norm_performance), 
            0, 1, color = "b", linestyle = "dashed", 
            label = "bbox mean")

plt.legend(loc = "best")     