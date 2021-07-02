# Importing the Libraries #


import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression

import warnings

warnings.filterwarnings( 'ignore' )





# Importing and Reading the Data #


Training_Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project Dataset/Boston House Price Prediction/Training Dataset.csv" )

Testing_Dataset = pd.read_csv( "C:/Users/Vaibhavi Nayak/Desktop/Project Dataset/Boston House Price Prediction/Testing Dataset.csv" )





# Cleaning the Data #


# 1. Handling the Missing Values #


# We would be removing the columns with more than 70% Missing Values in either the Training Dataset or the Testing Dataset. #


Training_Dataset = Training_Dataset.dropna( thresh = 0.70 *len( Training_Dataset ) , axis = 1 ) 

Testing_Dataset = Testing_Dataset.dropna( thresh = 0.70 *len( Testing_Dataset ) , axis = 1 ) 


# We would be filling the columns with less than 70 % Missing Values in either the Training Dataset or the Testing Dataset with the Mean of the Column. #


Training_Dataset = Training_Dataset.fillna( Training_Dataset.mean() )

Testing_Dataset = Testing_Dataset.fillna( Testing_Dataset.mean() )



# 2. Correlation Matrix #


Correlation_Matrix = Training_Dataset.corr().round( 2 ) 

print( Correlation_Matrix )


# sns.heatmap( data = Correlation_Matrix )





# Feature Variables #


X = pd.DataFrame( np.c_[ Training_Dataset[ 'LSTAT' ] , Training_Dataset[ 'RM' ] , Training_Dataset[ 'PTRATIO' ] ] , columns = [ 'LSTAT' , 'RM' , 'PTRATIO' ] )


# Target Variables #


Y = Training_Dataset[ 'MEDV' ]





# Splitting the Data for Training the Dataset and Testing the Dataset #


from sklearn.model_selection import train_test_split


X_Training_Dataset , X_Testing_Dataset , Y_Training_Dataset , Y_Testing_Dataset = train_test_split( X , Y , test_size = 0.2 , random_state = 5 )





# Training the Model #


Model = LinearRegression()

Model.fit( X_Training_Dataset , Y_Training_Dataset )





# Accuracy of the Training Dataset #


Accuracy_Training_Dataset = Model.score( X_Training_Dataset , Y_Training_Dataset ) * 100

print( Accuracy_Training_Dataset )


# Accuracy of the Testing Dataset #


Accuracy_Testing_Dataset = Model.score( X_Testing_Dataset , Y_Testing_Dataset ) * 100

print( Accuracy_Testing_Dataset )


# Accuracy of the Model #


from sklearn.metrics import r2_score

Model_Accuracy = r2_score( Y , Model.predict( X ) ) * 100

print( Model_Accuracy )


























