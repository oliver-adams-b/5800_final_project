import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,8)

def generate_cluster(x, y, var, count):
    dx = np.random.normal(0, var, count)
    dy = np.random.normal(0, var, count)
    return np.array([dx, dy]).reshape((count, 2)) + [x, y]
   
def generate_data(means, stds, count):
    clusters = [generate_cluster(x[0], x[1], s, count) for x, s in zip(means, stds)]
    data = np.append(clusters[0], clusters[1], axis = 0)
    
    for i in range(2, len(means)):
        data = np.append(data, clusters[i], axis = 0)
    
    return data

def disp_data(cluster, 
              color = "b"):
    plt.scatter(cluster[:, 0], 
                cluster[:, 1], 
                c = color, 
                s = 6, 
                alpha = 0.5)
    
def l2(x, y):
    return np.linalg.norm(x - y)

def find_nearest_centroid(point, centroids):
    return np.argmin([l2(point, c) for c in centroids])

def get_bbox_for_data(data):
    #data is an array of points
    maxx, maxy = np.array([np.max(data[:, 0]), np.max(data[:, 1])])
    minx, miny = np.array([np.min(data[:, 0]), np.min(data[:, 1])])
    return minx, miny, maxx, maxy

def generate_initial_centroids_from_bbox(bbox, k):
    centroids = []
    minx, miny, maxx, maxy = bbox
    for _ in range(k):
        x = np.random.randint(minx, maxx)
        y = np.random.randint(miny, maxy)
        centroids.append([x, y])
    return np.asarray(centroids)

def generate_initial_centroids_from_data(data, k):
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

    init_centroids = generate_initial_centroids_from_data(data, k)
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
    
    bins = [[c] for c in centroids]
    
    for d in data:
        bins[find_nearest_centroid(d, centroids)].append(d.tolist())
      
    if disp:
        colors = ["g", "b", "c", "m", "y", "k"]
        for c in bins:
            l = colors[np.random.randint(len(colors))]
            disp_data(np.array(c), color = l)
            colors.remove(l)
    
    return bins


#generate some test clusters
k = 6

#randomly generate the locations of the random clusters, a set of k points on the plane
means = np.random.randint(-100, 100, (k, 2))
#randombly generate the variance of the cluster surrounding each point
stds = np.random.randint(1, 20, (k,))

#now generate the test data from the locations and variances:
data = generate_data(means,
                     stds, 
                     500)


#run k-means:
centroids = k_means(data, k, disp = False)

#get the classification bins for the data
class_bins = classify_from_centroids(data, centroids)


#plot the results:
for i in range(k):
    plt.scatter(centroids[i][0], centroids[i][1], s = 100, c = "r")
    
plt.axis("off")
plt.title("K-means clustering of {} random Normal Distributions".format(k))
plt.show()


