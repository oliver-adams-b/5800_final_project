import numpy as np
import matplotlib.pyplot as plt


def generate_cluster(x, y, var, count):
    dx = np.random.normal(0, var, count)
    dy = np.random.normal(0, var, count)
    return np.array([dx, dy]).reshape((count, 2)) + [x, y]
   
def generate_data(means, count):
    clusters = [generate_cluster(x, y, 10, count) for x, y in means]
    data = np.append(clusters[0], clusters[1], axis = 0)
    
    for i in range(2, len(means)):
        data = np.append(data, clusters[i], axis = 0)
    
    return data

def disp_data(cluster):
    plt.scatter(cluster[:, 0], 
                cluster[:, 1], 
                c = "b", 
                s = 2, 
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
            init_centroids, 
            max_num_steps, 
            min_delta = 0.01):
    
    prev_centroids = init_centroids
    centroids = init_centroids
    k = len(init_centroids)
    
    
    for _ in range(max_num_steps):
        
        prev_centroids = centroids
        bins = [[c] for c in centroids]
        
        for p in data:
            bins[find_nearest_centroid(p, centroids)].append(p)  
        centroids = np.array([np.mean(d, axis = 0) for d in bins])
        disp_data(data)
        
        
        for i in range(k):
            plt.scatter(centroids[i][0], centroids[i][1], s = 100, c = "r")

        plt.axis("off")
        plt.show()
        
        delta = np.linalg.norm(np.sum(prev_centroids-centroids, axis = 0))
    
        if delta < min_delta:
            print("min delta reached: {}".format(delta))
            break
        
    return centroids
    
#generate some test clusters
k = 4
means = np.random.randint(-100, 100, (k, 2))

data = generate_data(means, 200)
disp_data(data)

#now display the bounding box, where we initialize the initial centroids
minx, miny, maxx, maxy = get_bbox_for_data(data)
plt.hlines([miny, maxy], minx, maxx)
plt.vlines([minx, maxx], miny, maxy)

#centroids = generate_initial_centroids_from_bbox([minx, miny, maxx, maxy], k)
centroids = generate_initial_centroids_from_data(data, k)

for i in range(k):
            plt.scatter(centroids[i][0], centroids[i][1], s = 100)

plt.show()
#
k_means(data, centroids, 100)