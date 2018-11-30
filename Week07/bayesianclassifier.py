import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# Set configuration
K = 3       # number of classes
N = 100     # number of train samples per class
Nt = 10000  # number of test samples per class

# Generate data & Save into a npz file (do once)
m1 = np.mat([0, 0]);    s1 = np.mat([[1, 0], [0, 1]])
m2 = np.mat([3, 3]);    s2 = np.mat([[1, 1.6], [1.6, 4]])
m3 = np.mat([0, 3.5]);  s3 = np.mat([[2, 0], [0, 1]])
X1 = np.matmul(np.random.randn(N, 2), sqrtm(s1)) + np.tile(m1, (N, 1))
X2 = np.matmul(np.random.randn(N, 2), sqrtm(s2)) + np.tile(m2, (N, 1))
X3 = np.matmul(np.random.randn(N, 2), sqrtm(s3)) + np.tile(m3, (N, 1))
Xt1 = np.matmul(np.random.randn(Nt, 2), sqrtm(s1)) + np.tile(m1, (Nt, 1))
Xt2 = np.matmul(np.random.randn(Nt, 2), sqrtm(s2)) + np.tile(m2, (Nt, 1))
Xt3 = np.matmul(np.random.randn(Nt, 2), sqrtm(s3)) + np.tile(m3, (Nt, 1))
np.savez('bayes_data.npz', X1=X1, X2=X2, X3=X3, Xt1=Xt1, Xt2=Xt2, Xt3=Xt3)

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

plt.scatter(np.array(Xt1[:,0]), np.array(Xt1[:,1]), c='r', label='Class 1', alpha=0.5)
plt.scatter(np.array(Xt2[:,0]), np.array(Xt2[:,1]), c='g', label='Class 2', alpha=0.5)
plt.scatter(np.array(Xt3[:,0]), np.array(Xt3[:,1]), c='b', label='Class 3', alpha=0.5)
plt.axis([-4, 6, -4, 8])
plt.legend()
plt.grid()
plt.title('Scatter Plot of Test')
plt.show()

# # Bayesian Classifier
# Calculate some parameters
M = np.mat([np.mean(X1, axis=0), np.mean(X2, axis=0), np.mean(X3, axis=0)])
S = np.array([np.cov(X1.T), np.cov(X2.T), np.cov(X3.T)])
Smean = (np.cov(X1.T) + np.cov(X2.T) + np.cov(X3.T)) / 3
Dtrain = np.concatenate((X1, X2, X3))
Etrain = np.zeros((3,1))
Dtest = np.concatenate((Xt1, Xt2, Xt3))
Etest = np.zeros((3,1))
d1 = np.zeros((3,1)); d2 = np.zeros((3,1)); d3 = np.zeros((3,1))

# Compute training error
for i in range(K):
    X = Dtrain[i*N:(i+1)*N,:]
    for j in range(N):
        for k in range(K):
            # Common Identity Covariance Matrix
            d1[k,0] = (X[j,:] - M[k,:]) * (X[j,:] - M[k,:]).T
            # Common Covariance Matrix
            d2[k,0] = (X[j,:] - M[k,:]) * np.linalg.inv(Smean) * (X[j,:] - M[k,:]).T
            # General Covariance Matrix
            d3[k,0] = (X[j,:] - M[k,:]) * np.linalg.inv(S[k,:]) * (X[j,:] - M[k,:]).T
        Yhat1 = np.argmin(d1)
        if (Yhat1 != i):
            Etrain[0,0] += 1
        Yhat2 = np.argmin(d2)
        if (Yhat2 != i):
            Etrain[1,0] += 1
        Yhat3 = np.argmin(d3)
        if (Yhat3 != i):
            Etrain[2,0] += 1
Error_rate_train = Etrain / (N*K)
print('Overall Training Error Rate')
print('Type 1: Same Identity Covariance Matrix\t {0:6.2f}%'.format(Error_rate_train[0,0]*100))
print('Type 2: Same Covariance Matrix\t\t\t {0:6.2f}%'.format(Error_rate_train[1,0]*100))
print('Type 3: General Covariance Matrix\t\t {0:6.2f}%'.format(Error_rate_train[2,0]*100))
print('')

# Compute test error
for i in range(K):
    X = Dtest[i*Nt:(i+1)*Nt,:]
    for j in range(Nt):
        for k in range(K):
            # Common Identity Covariance Matrix
            d1[k,0] = (X[j,:] - M[k,:]) * (X[j,:] - M[k,:]).T
            # Common Covariance Matrix
            d2[k,0] = (X[j,:] - M[k,:]) * np.linalg.inv(Smean) * (X[j,:] - M[k,:]).T
            # General Covariance Matrix
            d3[k,0] = (X[j,:] - M[k,:]) * np.linalg.inv(S[k,:]) * (X[j,:] - M[k,:]).T
        Yhat1 = np.argmin(d1)
        if (Yhat1 != i):
            Etest[0,0] += 1
        Yhat2 = np.argmin(d2)
        if (Yhat2 != i):
            Etest[1,0] += 1
        Yhat3 = np.argmin(d3)
        if (Yhat3 != i):
            Etest[2,0] += 1
Error_rate_test = Etest / (Nt*K)
print('Overall Test Error Rate')
print('Type 1: Same Identity Covariance Matrix\t {0:6.2f}%'.format(Error_rate_test[0,0]*100))
print('Type 2: Same Covariance Matrix\t\t\t {0:6.2f}%'.format(Error_rate_test[1,0]*100))
print('Type 3: General Covariance Matrix\t\t {0:6.2f}%'.format(Error_rate_test[2,0]*100))