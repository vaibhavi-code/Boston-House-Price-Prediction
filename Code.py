############################################################################################################################################################################################


# Importing the Libraries #


import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression


############################################################################################################################################################################################


# Ignoring the Warnings #

import warnings

warnings.filterwarnings( 'ignore' )


############################################################################################################################################################################################


# Importing the Data #


Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project/3. Boston House Price Prediction/Dataset.csv" )


############################################################################################################################################################################################


# Cleaning the Data #


# 1. Handling the Missing Values #


Dataset = Dataset.dropna( thresh = 0.70 *len( Dataset ) , axis = 1 ) 

Dataset = Dataset.fillna( Dataset.mean() )


############################################################################################################################################################################################


# Normalization #


for column in Dataset.columns:

    Dataset[ column ] = Dataset[ column ] / Dataset[ column ].max()


############################################################################################################################################################################################


# Feature Selection #


# Splitting the Dataset #

y = Dataset[ 'MEDV' ]

Columns = [ 'MEDV' ]

x = Dataset.drop( Columns , axis = 1 )

from sklearn.model_selection import train_test_split

X_Training_Dataset , X_Testing_Dataset , Y_Training_Dataset , Y_Testing_Dataset = train_test_split( x , y , test_size = 0.2 , random_state = 0 )


# Selecting the Features #

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

selected_features = SelectKBest( score_func = f_regression , k = 7 ).fit( X_Training_Dataset , Y_Training_Dataset )


# print( 'Score List : ' , selected_features.scores_ )

# print( 'Feature List : ' , X_Training_Dataset.columns )


X_Training_Dataset = selected_features.transform( X_Training_Dataset )

X_Testing_Dataset = selected_features.transform( X_Testing_Dataset )


############################################################################################################################################################################################


# Training the Model and Testing the Model #


# Linear Regression #

from sklearn.linear_model import LinearRegression

Model_1 = LinearRegression()

Model_1.fit( X_Training_Dataset , Y_Training_Dataset )

Prediction_1 = Model_1.predict( X_Testing_Dataset )


# Random Forest #

from sklearn.ensemble import RandomForestRegressor

Model_2 = RandomForestRegressor()

Model_2.fit( X_Training_Dataset , Y_Training_Dataset )

Prediction_2 = Model_2.predict( X_Testing_Dataset )


############################################################################################################################################################################################


# Accuracy #


from sklearn.metrics import r2_score


Accuracy_1 = r2_score( Y_Testing_Dataset , Prediction_1 )

print( 'Linear Regression' , Accuracy_1 )


Accuracy_2 = r2_score( Y_Testing_Dataset , Prediction_2 )

print( 'Random Forest' , Accuracy_2 )


############################################################################################################################################################################################

