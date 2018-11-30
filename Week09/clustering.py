import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# # Generate data & Save into a npz file
# Set configuration
K = 3      # number of clusters
N = 100     # number of train samples per class
# once
X = np.concatenate((np.random.randn(N, 2) + np.array([0,0]),
                    np.random.randn(N, 2) + np.array([3, 0]),
                    np.random.randn(N, 2) + np.array([1.5, 3])), axis=0)
np.savez('cluster_data.npz', X=X)

# Load data from a npz file
npzfile = np.load('cluster_data.npz', )
X = npzfile['X']

# Configuration parameters
K = 3   # number of clusters
N = X.shape[0]  # number of data
C = X.shape[1]  # number of features

# Generate random centers
mean = np.mean(X, axis=0); std = np.std(X, axis=0)
centers = np.random.randn(K, C)*std+mean

# Scatter plot (Initial)
colors = ['r', 'g', 'b']
plt.scatter(X[:,0], X[:,1], c='black', s=7, label = 'Data')
plt.scatter(centers[:,0], centers[:,1], marker='s', s=50, c='m', label ='Center')
plt.title('Raw Data with Random Centers')
plt.xlabel('X1'); plt.ylabel('Y1'); plt.legend()
plt.show()

# Set the parameters for K-means clustering
centers_old = np.zeros(centers.shape)   # old centers
centers_new = deepcopy(centers)         # new centers
clusters = np.zeros(N)
distances = np.zeros((N,K))
error = np.linalg.norm(centers_new - centers_old)

# K-means clustering loop
iter = 1
while error != 0:
    # calculate the distance between centers and data
    for i in range(K):
        distances[:,i] = np.linalg.norm(X-centers_new[i], axis=1)
    # assign data to closest center
    clusters = np.argmin(distances, axis=1)
    # keep old center position
    centers_old = deepcopy(centers_new)
    # calculate new centers
    for i in range(K):
        centers_new[i] = np.mean(X[clusters==i], axis=0)
        plt.scatter(X[clusters==i,0], X[clusters==i,1], s=7, c=colors[i])
        plt.scatter(centers_new[i,0], centers_new[i,1], marker='s', s=50, c='m')
    error = np.linalg.norm(centers_new - centers_old)
    plt.title('K-means (iter={}, err={:.4f})'.format(iter, error))
    plt.xlabel('X1'); plt.ylabel('Y1');
    plt.show()
    print('Iteration = {}, Error = {:.4f}'.format(iter, error))
    iter += 1
print('Final Center Location')
print(centers_new)
