# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:44:19 2022

@author: Hp
"""

import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np

from sklearn import datasets, linear_model

datasets = datasets.load_diabetes()

df = pd.DataFrame(data = np.c_[datasets["data"] ,datasets["target"]])    #hum ny data set main sy data or target waly function liye or usko Data frame k ander add kr diye

df = df.rename(columns={0:"data0" , 1:"data1",  2:"data2" , 3:"data3"  , 4:"data4"  , 5:"data5" , 6:"data6" , 7:"data7"  , 8:"data8"  , 9:"data9" , 10 : "target"})        #hamary pas ho diabitites ka data tha us main 0-9 index tk column main indexing thi to hum ny us main column mk name enter kr diye


#Now let us define our x and y values for the model.
#x values will be time column, so we can define it by dropping cells
#x can be multiple independent variables which we will discuss in a different tutorial
#this is why it is better to drop the unwanted columns rather than picking the wanted column
#y will be cells column, dependent variable that we are trying to predict. 


X = df.drop("target" , axis=1)    #Hum ny features alag kr liye target ko drop kr k Q k target wala column hum ny as a label use krna tha

Y = df["target"]


from sklearn.preprocessing import MinMaxScaler        
from sklearn.preprocessing import QuantileTransformer

scaler = MinMaxScaler()          
scaler.fit(X)    
X = scaler.transform(X) 

from sklearn.model_selection import train_test_split

#SPlit data into training and test datasets so we can validate the model using test data
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2 , random_state=44)       #yahan hum ny randomly 20 percent data testing k liye alag kiya ko testing training data ko un variables main store kr diya

#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time


model = linear_model.LinearRegression()      #hum ny linear regression k object model name k variable main store krwaya Q k hum ny apny data pr linear regression apply krni thi
model.fit(x_train, y_train)      #traingig data or labael apny model main fitt kiye

#For linear regression, Y=the value we want to predict
#X= all independent variables upon which Y depends. 
#3 steps for linear regression....
#Step 1: Create the instance of the model
#Step 2: .fit() to train the model or fit a linear model
#Step 3: .predict() to predict Y for given X values. 
# Step 4: Calculate the accuracy of the model. 


print("score   ",model.score(x_train, y_train))     ##Prints the R^2 value, a measure of how well   #yeh hamain yeh btata ha k jo model hum train kiya ha data dy kr wo kitna percent train howa ha matlab kitna acha train howa ha

prediction = model.predict(x_test)    #yahan or features apny model ko dy kr test krty hain k kitna resullt kya aarhy hain kya predict kr rha ha hamara model
from sklearn import metrics

# A MSE value of about 8 is not bad compared to average # cells about 250.
#print("Mean sq. errror between y_test and predicted =", np.mean(prediction-y_test)**2)    #yeh mean square error find kr raha ha