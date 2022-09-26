# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:34:03 2021

@author: Lenovo
"""

#import numpy as np
import pandas as pd
import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

# printing the stopwords
print(stopwords.words('english'))

# load the data to a pandas dataframe
news_data = pd.read_csv('train.csv')

# first 5 rows of the dataset
news_data.head()

"""
     0 --> Real News

    1 --> Fake News
"""

print(news_data.shape)

# checking for missing values
print(news_data.isnull().sum())

# replacing the missing values with null string
news_data = news_data.fillna('')

# merging th eauthor name and news title
news_data['content'] = news_data['author']+' '+news_data['title']

# first 5 rows of the dataset
news_data.head()

# separating feature and target
X = news_data.drop(columns='label', axis =1)
Y = news_data['label']

print(X)
print(Y)

#Stemming is the process of reducing a word to its Root Word
port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

news_data['content'] = news_data['content'].apply(stemming)
print(news_data['content'])

X = news_data['content'].values
Y = news_data['label'].values

print(X)
print(Y)

Y.shape

# converting the textual data to feature vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print(X)