import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from pandas.plotting import scatter_matrix


'''
Only once
'''
# Querying Iris dataset from UCI Machine Learning repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
# Save Iris dataset to a csv file
dataset.to_csv('iris.csv', index=False)


'''
Load data
'''
dataset = pd.read_csv('./iris.csv')

# See first 5 elements
print(dataset.head())


'''
Summary
'''
print('\nInfo---------------------')
print(dataset.info())

print('\nDescriptions-------------')
print(dataset.describe())

print('\nClass Distribution-------')
print(dataset.groupby('class').size())


'''
Visualization
'''
# box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.plot(kind='box', sharex=False, sharey=False)
dataset.boxplot(by='class', figsize=(10,10), grid=False)
plt.show()

# Violin plot
sns.violinplot(data=dataset, x='class', y='sepal-length')
plt.title('Violin Plot of Iris Dataset')
plt.show()

# histogram
dataset.hist(edgecolor='black', linewidth=1.2, grid=False)
plt.show()

# Scatter plot using pandas
scatter_matrix(dataset, figsize=(10,10)) # Using pandas
plt.show()

# Scatter plot using seaborn
sns.pairplot(dataset, hue='class')  # Using seaborn
sns.pairplot(dataset, hue='class', diag_kind='kde')
plt.show()