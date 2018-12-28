import numpy as np
import matplotlib.pyplot as plt

# tanh 1차 도함수
def d_tanh(a):
    return np.multiply(1-a, 1+a)

# MLP 이용한 오차계산 함수
def MLPtest(Xtst, Ytst, w, wo, v, v0):
    N = Xtst.shape[0]
    E = np.zeros((N, 1))
    Yhat = np.zeros((N, 2))
    for i in range(N):
        x = Xtst[i,:].reshape(1,2)
        t = Ytst[i,:].reshape(1,2)
        uh = np.matmul(x, w) + w0
        z = np.tanh(uh)
        uo = np.matmul(z, v) + v0
        y = np.tanh(uo)
        e = y - t
        E[i,] = np.matmul(e, e.T)
        if y[0,0]>y[0,1]:
            Yhat[i,:] = [1, -1]
        else:
            Yhat[i,:] = [-1, 1]
    SEtst = np.sum(np.sqrt(E)) / N
    diffTY = np.sum(np.abs(Ytst[:,0]-Yhat[:,0])) / 2
    CEtst = diffTY/N * 100
    return (SEtst, CEtst)

# Load data
Train = np.genfromtxt('NN_TRAIN.csv', delimiter=',')
Test = np.genfromtxt('NN_TEST.csv', delimiter=',')
X = Train[:,:2]
Y = Train[:,2:]
Xt = Test[:,:2]
Yt = Test[:,2:]

# Set parameters
N = X.shape[0]
INP = 2; HID = 3; OUT = 2
w = np.random.rand(INP, HID) * 0.4 - 0.2
w0 = np.random.rand(1, HID) * 0.4 - 0.2
v = np.random.rand(HID, OUT) * 0.4 - 0.2
v0 = np.random.rand(1, OUT) * 0.4 - 0.2
eta = 0.001
Mstep = 1000
Elimit = 0.1
E = np.zeros((N,1))
Serr = np.zeros((Mstep,1))
Cerr = np.zeros((Mstep,1))
Accu = np.zeros((Mstep,1))

# MLP Training
for epoch in range(Mstep):
    for i in range(N):
        x = X[i,:].reshape(1,2)
        t = Y[i,:].reshape(1,2)
        uh = np.matmul(x, w) + w0
        z = np.tanh(uh)
        uo = np.matmul(z, v) + v0
        y = np.tanh(uo)
        e = y - t
        E[i,] = np.matmul(e, e.T)
        delta_v = np.multiply(d_tanh(y), e)
        delta_w = np.multiply(d_tanh(z), (np.matmul(delta_v, v.T)))
        v = v - eta * np.matmul(z.T, delta_v)
        v0 = v0 - eta * delta_v
        w = w - eta * np.matmul(x.T, delta_w)
        w0 = w0 - eta * delta_w
    serr, cerr = MLPtest(X, Y, w, w0, v, v0)
    print('Epoch: {}  MSE: {:7.4f}  Accuracy: {:7.2f}'.format(epoch+1, serr, 100-cerr))
    Serr[epoch,] = serr
    Cerr[epoch,] = cerr
    Accu[epoch,] = 100-cerr
    if serr < Elimit:
        break

print(v)
print(v0)
print(w)
print(w0)
# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Serr, label='SMSE')
ax2 = ax.twinx()
ax2.plot(Accu, color='r', label='Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Sum of Mean Squared Error')
ax2.set_ylabel('Accuracy (%)')
fig.legend()
plt.show()