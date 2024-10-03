# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

import matplotlib.pyplot as plt

# %% [markdown]
# # k-nearest neighbors
# 
# This dataset was obtained from https://archive.ics.uci.edu/ml/datasets/Heart+Disease (this is a great resource for datasets to try machine learning on). It has data on patients that are and are not diagnosed with heart disease.
# 
# The attributes are:
# * age: age in years 
# * sex: sex (1 = male; 0 = female) 
# * cp: chest pain type 
#  * -- Value 1: typical angina 
#  * -- Value 2: atypical angina 
#  * -- Value 3: non-anginal pain 
#  * -- Value 4: asymptomatic 
# * trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
# * chol: serum cholestoral in mg/dl 
# * fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
# * restecg: resting electrocardiographic results 
#  * -- Value 0: normal 
#  * -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
#  * -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
# * thalach: maximum heart rate achieved 
# * exang: exercise induced angina (1 = yes; 0 = no) 
# * oldpeak = ST depression induced by exercise relative to rest 
# * slope: the slope of the peak exercise ST segment 
#  * -- Value 1: upsloping 
#  * -- Value 2: flat 
#  * -- Value 3: downsloping 
# * ca: number of major vessels (0-3) colored by flourosopy 
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
# * num: diagnosis of heart disease (angiographic disease status) 
#  * -- Value 0: absence.
#  * -- Value 1,2,3,4: presence of heart disease
# 
# %% [markdown]
# # Explore the data
# 
# Read in the data, modify the dependent variable name and plot a histogram of the ages of patients, both healthy and those with heart disease.

# %%
# Comma-separated values
df = pd.read_csv('cleveland.csv')

# Rename 'num' column to 'disease' and change 1,2,3,4 to 1
df = df.rename({'num':'disease'}, axis=1)
df['disease'] = df.disease.apply(lambda x: min(x, 1))
display(df.head(5))

# Plot histograms
fig, (ax1, ax2) = plt.subplots(2, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

ax1.hist(df[df.disease == 0].age);
# ax1.set_xlabel('age');
ax1.set_ylabel('number of patients');
ax1.set_xlim(20, 80);
ax1.set_ylim(0, 50);
ax1.set_title('healthy');

ax2.hist(df[df.disease == 1].age, color='orange');
ax2.set_xlabel('age');
ax2.set_ylabel('number of patients');
ax2.set_xlim(20, 80);
ax2.set_ylim(0, 50);
ax2.set_title('has heart disease');

# %% [markdown]
# # k-nearest neighbors - first try
# 
# Try it first on age, using the scikit-learn package. This code simply looks for the five patients with ages closest to a given age, then prints how many of those patients are healthy and how many have heart disease.

# %%
# Use knn on age. First create a nearest neighbors object.
nn = NearestNeighbors(n_neighbors=8, metric='euclidean', algorithm='auto')

# Create a two-dimensional array. This is basically a one-dimensional array with
# single-element arrays of patient ages in the second dimension. We're going to
# search for neighbors using only the age dimension.
X = [[x] for x in df.age]

# This builds an index data structure under the hood for query performance
fit = nn.fit(X)

# Find the k nearest neighbors
distances, indices = fit.kneighbors([[65]])
display(indices[0])

# Get the patients that are near the age
nbrs = df.iloc[indices[0]]
display(nbrs)

# Print how many patients are sick and how many are healthy
healthy = nbrs[nbrs.disease == 0].count().disease
sick = nbrs[nbrs.disease == 1].count().disease
print('healthy: {}\nsick: {}'.format(healthy, sick))

# %% [markdown]
# # multiple dimensions
# 
# Now run knn on a patient from the database using an additional dimension, or attribute: trestbps (resting blood pressure).
# 
# **Warning** The data used in this example is not standardized, so differences in the magnitude of change between the different attributes could cause one attribute to unduly influence another.

# %%
df[['age', 'trestbps']].values


# %%
# df[['age', 'trestbps']].values
X = df[['age', 'trestbps']].values
y = df[['disease']].values

# This builds an index data structure under the hood for query performance
fit = nn.fit(X)

# Get a random patient to test on
i = random.randint(0,len(X)-1)
patientX = X[i]
patienty = y[i]
display(df.iloc[i])

# Find the k nearest neighbors to the patient. Problem: the patient
# itself will be found in the list of neighbors!
distances, indices = fit.kneighbors([patientX])
nbrs = df.iloc[indices[0]]
display(nbrs)

healthy = nbrs[nbrs.disease == 0].count().disease
sick = nbrs[nbrs.disease == 1].count().disease
print('healthy: {}\nsick: {}'.format(healthy, sick))
predict = 0 if (healthy > sick) else 1
actual = 0 if (patienty == 0) else 1
success = predict == actual
print(success)

# %% [markdown]
# # multiple tests

# %%
X = df[['age', 'trestbps']].values
y = df[['disease']].values

# This builds an index data structure under the hood for query performance
fit = nn.fit(X)

# Get random patients to test on
n = 7
pindices = [random.randint(0,len(X)-1) for _ in range(n)]
patientsX = X[pindices]
patientsy = y[pindices]

# Find the k nearest neighbors to the patient. Problem: we still
# have the problem of the patient itself being found!
distances, indices = fit.kneighbors(patientsX)
print('indices of k-nearest neighbors for each patient:')
display(indices)

for i in range(n):
    print('nearest neighbors to patient: {}:'.format(patientsX[i]))
    nbrs = df.iloc[indices[i]]
    display(nbrs)

# This is where we would compile how many patients are predicted
# correctly.

# %% [markdown]
# # Split data into train/test and get precision/recall/f score

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

k = 20
nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')

X = df[['age', 'trestbps', 'chol', 'thalach']].values
y = df[['disease']].values

# Use random_state if you want the same values each time you run for debugging,
# but you should select the split randomly when you're ready to actually train
# and test on the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)#, random_state=42)

# Build the model with the training data and test with the test data

# You may also want to use this function to compute the f score. The labels parameter indicates
# that label 1 is "positive" in the sense of "true positive", "false positive" etc.
# (p,r,f,s) = precision_recall_fscore_support(y_test, y_pred, labels=[1])

# %% [markdown]
# # Curse of dimensionality
# "In low-dimensional datasets, the closest points tend to be much closer than average. But two points are close only if they're close in every dimension, and every extra dimension -- even if just noise -- is another opportunity for each point to be further away from every other point. When you have a lot of dimensions it's likely that the closest points aren't much closer than average, which means that two points being close doesn't mean very much (unless there is a *lot* of structure in your data)." -Joel Grus
# 
# The chart that this code displays shows that distance has less meaning as dimensions grows. In higher dimensions, most points are about the same distance from each other.

# %%
import random
import math

def random_point(dim):
    return [random.random() for _ in range(dim)]

def sq(x):
    return x*x

def distance(a, b):
    a = [sq(a[i]) + sq(b[i]) for i in range(len(a))]
    return math.sqrt(sum(a))

def random_distances(dim, num_pairs):
    return [distance(random_point(dim), random_point(dim)) for _ in range(num_pairs)]

print(random_point(3))
print(random_distances(3, 10))

num_pairs = 100
means = []
mins = []
for dim in range(1, 100):
    d = random_distances(dim, num_pairs)
    means.append(sum(d)/len(d))
    mins.append(min(d))

# Note that the ratio of average distance to min distance
# approaches one, so there's less space between the two.
plt.plot(range(1, 100), means, label='average distance')
plt.plot(range(1, 100), mins, label='min distance')
plt.xlabel('num dimensions')
plt.ylabel('distance')
plt.title('Distances of 100 randomly selected points')
plt.legend()


