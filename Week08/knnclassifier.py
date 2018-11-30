import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import  KNeighborsClassifier

# Load data from a npz file
npzfile = np.load('bayes_data.npz')
X1 = npzfile['X1']; X2 = npzfile['X2']; X3=npzfile['X3']
Xt1 = npzfile['Xt1']; Xt2 = npzfile['Xt2']; Xt3=npzfile['Xt3']

# Display scatter plot
plt.scatter(np.array(X1[:,0]), np.array(X1[:,1]), c='r', label='Class 1', alpha=0.5)
plt.scatter(np.array(X2[:,0]), np.array(X2[:,1]), c='g', label='Class 2', alpha=0.5)
plt.scatter(np.array(X3[:,0]), np.array(X3[:,1]), c='b', label='Class 3', alpha=0.5)
plt.axis([-4, 6, -4, 8])
plt.legend()
plt.grid()
plt.title('Scatter Plot of Train')
plt.show()

# # Nearest neighbor  Classifier
# Calculate some parameters
K = 1       # number of neighbors
N = 100     # number of train samples per class
Nt = 10000  # number of test samples per class
# Set ground truth (class label)
Y = np.concatenate((np.mat(np.ones((N, 1))),
                   np.mat(np.ones((N, 1)))*2,
                   np.mat(np.ones((N, 1)))*3))
Yt = np.concatenate((np.mat(np.ones((Nt, 1))),
                   np.mat(np.ones((Nt, 1)))*2,
                   np.mat(np.ones((Nt, 1)))*3))
Dtrain = np.concatenate((X1, X2, X3))
Dtest = np.concatenate((Xt1, Xt2, Xt3))
Etest = 0
distXt = np.zeros((len(Y),1))

print('Using Nearest Neighbor Classifier')
# # Computing distance between train and test
# Implementation 1-NN
for i in range(len(Yt)):
    X = Dtest[i,:]
    for j in range(len(Y)):
        distXt[j,0] = distance.euclidean(X, Dtrain[j,:])
    minIdx = np.argmin(distXt)
    if Y[minIdx,0] != Yt[i,0]:
        Etest += 1
Error_rate_test = Etest / (len(Yt))
print('Overall Test Error Rate {0:6.2f}%'.format(Error_rate_test*100))

# Using sklearn knn
for k in [1, 5, 10, 50, 100]:
    neigh = KNeighborsClassifier(n_neighbors=k)  # Set K-NN
    neigh.fit(Dtrain, np.ravel(Y))  # Train K-NN
    # YtHat = neigh.predict(Dtest)              # Predict class label
    accu = neigh.score(Dtest, np.ravel(Yt))  # Compute accuracy
    print('Overall Test Error Rate by sklearn {0:6.2f}% (K={1:})'.format((1 - accu) * 100, k))