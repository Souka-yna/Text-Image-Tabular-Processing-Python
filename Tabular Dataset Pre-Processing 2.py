# -*- coding: utf-8 -*-
"""
Created on Mon May 17 23:22:25 2021

@author: Lenovo
"""
# baseline model performance on the wine dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/wine.csv'
df = read_csv(url, header=None)
data = df.values
X, y = data[:, :-1], data[:, -1]

# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))

# define the model
model = LogisticRegression(solver='liblinear')

# define the cross-validation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
