import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)



def generate_cluster(mean, cov, count):
    """creates a cluster of points with provided mean (x, y) and variance var"""
    return np.random.multivariate_normal(mean, cov, count)
   
def generate_data(means, covs, count):
    """creates a test-dataset from a list of means and standard deviations"""
    clusters = [generate_cluster(m, c, count) for m, c in zip(means, stds)]
    data = np.append(clusters[0], clusters[1], axis = 0)
    
    for i in range(2, len(means)):
        data = np.append(data, clusters[i], axis = 0)
    
    return data

def disp_data(cluster, 
              color = "b"):
    """just displays a data cluster, a list of points. """
    plt.scatter(cluster[:, 0], 
                cluster[:, 1], 
                c = color, 
                s = 6, 
                alpha = 0.5)
    
def l2(x, y):
    return np.linalg.norm(x - y)

def find_nearest_centroid(point, centroids):
    """returns the index of the nearest centroid for a provided point"""
    return np.argmin([l2(point, c) for c in centroids])

def generate_initial_centroids_from_data(data, k):
    """
    Randomly samples the data, which is just a list of points, and it does so
    k times and records the mean for each subsample. 
    
    Returns a list of k points which act as our first guess for the centroids. 
    """
    centroids = []
    m = int(len(data)/k**2)
    for _ in range(k):
        centroids.append(np.mean(data[np.random.choice(range(len(data)), (m))], axis = 0))
    return np.asarray(centroids)

def k_means(data, 
            k,
            max_num_steps = 100, 
            min_delta = 0.001, 
            disp = True):

    """
    Generates k centroids which best approximate the provided dataset. 
    """
    
    """creates our initial guess for the location of the centroids"""
    init_centroids = generate_initial_centroids_from_data(data, k)
    
    """keeps track of the location of the centroids for the previous iter, so that we
    can measure convergence between subsequent iterations"""
    prev_centroids = init_centroids 
    
    centroids = init_centroids
    
    for _ in range(max_num_steps):
        
        prev_centroids = centroids
        bins = [[c] for c in centroids]
        
        
        for p in data:
            bins[find_nearest_centroid(p, centroids)].append(p)  
        centroids = np.array([np.mean(d, axis = 0) for d in bins])

            
        if disp:
            for i in range(k):
                plt.scatter(centroids[i][0], centroids[i][1], s = 100, c = "r")
            disp_data(data)
            plt.axis("off")
            plt.show()
        
        delta = np.linalg.norm(np.sum(prev_centroids-centroids, axis = 0))
    
        if delta < min_delta:
            print("min delta reached: {}".format(delta))
            break
        
    return centroids
  
    
def classify_from_centroids(data, 
                            centroids, 
                            disp = True):
    
    """bins points into groups depending on their nearest centroid"""
    bins = [[c] for c in centroids]
    
    for d in data:
        bins[find_nearest_centroid(d, centroids)].append(d.tolist())
      
    if disp:
        colors = ["g", "b", "c", "m", "y", "k"]
        for c in bins:
            l = colors[np.random.randint(len(colors))]
            disp_data(np.array(c), color = l)
            #colors.remove(l)
    
    return bins


if __name__ == "__main__":
    """
    Below shows the example usage of the methods in this script:
        
        we first create a test dataset comprising of k clusters in the plane
        all with randomized mean and variance
        
        we then run k-means clustering on the dataset to get the locations 
        of centroids which best "cluster" the dataset. 
        
        then we plot the classificaiton of clusters as determined from the centroids
    """
    #generate k test clusters in d-dimensional space:
    k = 5
    d = 3
    #randomly generate the locations of the random clusters, a set of k points on the plane
    means = np.random.randint(-100, 100, (k, d))
    
    #randombly generate the variance of the cluster surrounding each point
    stds = np.random.randint(1, 40, (k, d, d))
    
    #now generate the test data from the locations and variances:
    data = generate_data(means,
                         stds, 
                         500)


    #run k-means:
    centroids = k_means(data, k, disp = False)
    
    print(centroids)
    #get the classification bins for the data
    class_bins = classify_from_centroids(data, centroids)
    
    #plot the results:
    for i in range(k):
        plt.scatter(centroids[i][0], centroids[i][1], s = 100, c = "r")
        
    plt.axis("off")
    plt.title("K-means clustering of {} random Normal Distributions".format(k))
    plt.show()


