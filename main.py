#Basic Linear Regression Tutorial for Machine Learning Beginner
#Created By Siddhant Mishra
#We will try to create a model to predict stock price of
#Google in next 3 days
import numpy as np
import pandas as pd
import  quandl as qd 
import math
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
# Getting the Stock data from Quandl
print("Getting stock from Quandl .........")
googleStock = qd.get('WIKI/GOOGL')

# Printing un altered dataset
pd.set_option('display.max_columns',None)
print("\nGoogle Stock from Quandl: \n\n {}".format(googleStock.head()))

# Indexing to filter relevant data
features = ['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']
dataset = googleStock[features]
print("--------------------------------------------------------------------\n"*2)
print("Filtered Dataset: \n {}".format(dataset.head()))

# Modifying/Defining features for modelset (data cleaning + manipulation)
# We remove outliers here and also NAN and other bad data
modelset = dataset[['Adj. Close', 'Adj. Volume']]
modelset.loc[:,'Stock Variance'] = (dataset['Adj. High']-dataset['Adj. Close']) / dataset['Adj. Close']*100
modelset.loc[:,'Percentage Change'] = (dataset['Adj. Close']-dataset['Adj. Open']) / dataset['Adj. Open']*100
modelset = modelset.fillna(-999999)
#modelset.fillna(-999999, inplace=True)

print("--------------------------------------------------------------------\n"*2)
print("Model Set W/O Label :\n {}".format(modelset.head()))

# Define Label
predictionCol = 'Adj. Close'
print(len(modelset))
forecastShift = int(math.ceil(0.005*len(modelset)))
modelset['Label'] = modelset[predictionCol].shift(-forecastShift)
modelset.dropna(inplace=True)

#Define X and Y for Linear Equation
# X is going to be our features and Y is going to be Label
# so we find Y = Omega(X)
X = np.array(modelset.drop(['Label'],1))
Y = np.array(modelset['Label'])

# Feature Scale and do it over entire dataset
# We do this before dividing data to have uniform normalization equation
X = preprocessing.scale(X)
Y = np.array(modelset['Label'])

# Create Training and Testing Set
XTrain, XTest, YTrain, YTest = cross_validation.train_test_split(X, Y, test_size=0.15)

# Create and Train model
clf = LinearRegression()
clf.fit(XTrain, YTrain)

accuracy = clf.score(XTest, YTest)
print("\n\nModel Accuracy = {}".format(accuracy)) 
