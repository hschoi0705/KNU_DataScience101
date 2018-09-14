# -*- coding: utf-8 -*-
"""Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TB6V1Sp7URKePraPw0cTp74QaDlFIPIX

# Data Preprocessing Examples

"""


"""
Standardization or mean removal or variance scaling (Z-score)
"""
from sklearn import preprocessing
import numpy as np

print("Data " + '='*20)
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

print("\nBefore Standardization" + '-'*20)
print(X_train)

X_scaled = preprocessing.scale(X_train)
print("\nAfter Standardization" + '-'*20)
print(X_scaled)

print("\nmean and standard-deviation " + '='*20)
print("\nBefore Standardization" + '-'*20)
print("mean " + str(X_train.mean(axis=0)))
print("var  " + str(X_train.var(axis=0)))

print("\nAfter Standardization" + '-'*20)
print("mean " + str(X_scaled.mean(axis=0)))
print("var  " + str(X_scaled.var(axis=0)))


"""
Min-Max Normalization
"""
from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  4.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
print(X_train), print('')

# MinMax method 1
X_minmax2 = preprocessing.minmax_scale(X_train, axis=0)
print(X_minmax2), print('')

# MinMax method 2
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_train)
print(X_minmax)


"""
Normalized vector or Unit vector
"""
from sklearn import preprocessing

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
print(X), print('')

# norm: L2(Euclidean distance)
X_normalized = preprocessing.normalize(X, norm='l2')
print(X_normalized)


"""
Binarization
"""
from sklearn import preprocessing

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
print(X), print('')

binarizer = preprocessing.Binarizer(threshold=0)
X_bin = binarizer.transform(X)
print(X_bin)


"""
One-hot-encoder
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

# Category(string) -> Numeric -> One-hot-vector
color = ['red', 'green', 'blue']
print(color)
color_num = LabelEncoder().fit_transform(color)
print(color_num)
color_onehot = OneHotEncoder().fit_transform(color_num.reshape(-1,1)).toarray()
print(color_onehot), print('')

# Category(string) -> One-hot-vector
color = ['red', 'green', 'blue']
print(color)
color_onehot2 = LabelBinarizer().fit_transform(color)
print(color_onehot2)


"""
Missing value imputation
"""
import numpy as np
from sklearn.preprocessing import Imputer

X = np.array([[1, 2], [np.nan, 3], [7, 6]])
print(X)
print('')
X_imp = Imputer(missing_values='NaN', strategy='mean', verbose=0).fit_transform(X)
print(X_imp)


"""
Image Pre-processing
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing

img = Image.open('lena.png').convert('L')   # Read Image as gray-scale image
npimg = np.array(img)       # convert Image to numpy array
print(npimg.shape)
print(npimg)

# MinMax Normalization
img_norm = preprocessing.minmax_scale(np.float64(npimg).reshape(-1,1), ).reshape(512, 512)
print(img_norm)

# Threshold
img_thresh = preprocessing.binarize(img_norm, threshold=0.5)

print("Min value is {0} and Max value is {1} of RAW image".format(np.amin(npimg), np.amax(npimg)))
print("Min value is {0:.4f} and Max value is {1:.4f} of MinMax Normalized image".format(np.amin(img_norm), np.amax(img_norm)))

f, ax = plt.subplots(1, 3)
ax[0].imshow(npimg, cmap='gray')
ax[1].imshow(img_norm, cmap='gray')
ax[2].imshow(img_thresh, cmap='binary')
plt.show()

# # Another Image Loading
# import matplotlib.image as mpimg
#
# img = mpimg.imread('lena.png')
# print(img)


"""
Read CSV file in order to make data feame(Pandas) AND Change column name
"""
import pandas as pd
# Read CSV file
dataset = pd.read_csv('iris.csv')

# Display attributes names
print(dataset.columns.values)
print('')

## Rename 'class' to 'species'
# method 1
new_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset.columns = new_names

# # method 2
# names = dataset.columns.tolist()
# names[names.index('class')] = 'species'
# dataset.columns = names
#
# # method 3
# dataset = dataset.rename(columns={'class': 'species'})
#
# # method 4
# dataset.rename(columns={'class': 'species'}, inplace=True)

# Display attributes names
print(dataset.columns.values)