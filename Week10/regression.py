import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
x = pd.Series([4, 8, 9, 8, 8, 12, 6, 10, 6, 9])
y = pd.Series([9, 20, 22, 15, 17, 30, 18, 25, 10, 20])


# Calculate each values and estimation parameters
n = x.size
sumy = y.sum(); avgy = y.mean()
sumx = x.sum(); avgx = x.mean()
sumxy = (x * y).sum()
sumsqrx = (x * x).sum()

print('n = {}'.format(n))
print('Sum y = {}, Average y = {}'.format(sumy, avgy))
print('Sum x = {}, Average x = {}'.format(sumx, avgx))
print('Sum of x*y = {}'.format(sumxy))
print('Sumf of square of x = {}'.format(sumsqrx))

b1 = (sumxy-(sumx*sumy/n))/(sumsqrx-(sumx*sumx)/10)
b0 = (sumy/n)-b1*(sumx/n)
print('\nSimple Linear Regression Result')
print('yhat = {:.3f}+{:.3f}x'.format(b0, b1))

# # Scatter plot
plt.plot(x, y, '.', color='red', label='Sample Data')

# define function
f = lambda x: b0 + b1*x # fitted function
xp = np.array([0, 13]) # using min, max points of x
plt.plot(xp, f(xp), color='blue', label='simple linear regression')

plt.xlabel('Cost')
plt.ylabel('Sales')
plt.legend()
plt.axis([0, 13, 0, 35])
plt.show()

# calculate r^2
yhat = f(x)
SST = ((y-avgy)*(y-avgy)).sum()
SSE = ((y-yhat)*(y-yhat)).sum()
SSR = ((yhat-avgy)*(yhat-avgy)).sum()
rsqr = SSR / SST
# rsqr = 1 - SSE/SST
print('\nSST = SSE + SSR')
print('{:.2f} = {:.2f} + {:.2f}'.format(SST, SSE, SSR))
print('R^2 = {:.3f}'.format(rsqr))