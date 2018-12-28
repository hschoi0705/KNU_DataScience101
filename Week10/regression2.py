import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data
x = np.array([4, 8, 9, 8, 8, 12, 6, 10, 6, 9])
y = np.array([9, 20, 22, 15, 17, 30, 18, 25, 10, 20])

# create linear regression object
regr = linear_model.LinearRegression()
# fitting
regr.fit(x.reshape(10,-1), y.reshape(10,-1))
# fitted regression line
yhat = regr.predict(x.reshape(10,-1))

# print result
b0 = regr.intercept_[0]
b1 = regr.coef_[0]
print('Simple Linear Regression Result using scikit-learn')
print('yhat = {:.3f}+{:.3f}x'.format((float)(b0), (float)(b1)))
print('R^2: {:.3f}'.format(r2_score(y, yhat)))

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