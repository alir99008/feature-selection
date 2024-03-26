# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:30:39 2022

@author: Hp
"""


#Random Forest Classiffer   : Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.
#for example, classifying whether an email is “spam” or “not spam”

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
 
df = pd.read_csv("breast cancer dataset.csv")

print(df.describe().T)  #Values need to be normalized before fitting. 


           #Null values hum es liye check krty hain Q k agr hamary data set main 1 B null value ho to usko hamra dataset deal nai krta
print(df.isnull().sum())                                  # yeh hamary har column main jitni null values honagi unko sum kr k tay ga
#df = df.dropna()

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis':'Label'})        #hum apny diagnosis waly column ko as a Label rakh k deal krain gy gy es liye hm eska name change kr k label rakh dain gyy
print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #M - malignant   B - benign              #Print graph of benin and malignant values

sns.distplot(df['radius_mean'], kde=False)    

print(df.corr())                    #yeh hamain har column ka corelation find kr k dy ga hamary  data main jitny B column howy unka subka

corrMatrix = df.corr()            
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
#sns.heatmap(df.iloc[:, 1:6:], annot=True, linewidths=.5, ax=ax)
sns.heatmap(corrMatrix, annot=False, linewidths=.5, ax=ax)              #yeh graph pehly find howy correlation ko dekhty 1 graph plot kr dy ga jo k bht zabardast define krta ha


#Replace categorical values with numbers
df['Label'].value_counts()              #yeh pehly B or M ko btay ga k total kitni bar aay ga

categories = {"B":1, "M":2}
df['Label'] = df['Label'].replace(categories)             #Hum B or M ko sirf es liye change krain gy Q k hamara data Sirf Numeric values pr he kam krta ha JAB k B or M Char hain...


#Define the dependent variable that needs to be predicted (labels)
Y = df["Label"].values                            #es main hum ny label ki values ko Y k ander store krwaya jo k hum apny model k as a label dain gyy

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["Label", "id"], axis=1)                   #es main hum ny label wala column es liye drop kiya Q k wo hum as a label use kr rhy thy es liye usko features main use nai kr sakty thy  or id wala column es liye drop kiya Q k uski koi zarorat nai thi
features_list = list(X.columns)  #List features so we can rank them later.
#from sklearn.preprocessing import normalize
#X = normalize(X, axis=1)



from sklearn.preprocessing import MinMaxScaler        
from sklearn.preprocessing import QuantileTransformer

#Min Max Scaler hamari bht bari or choti values ko 0 sy 1 k beach main change kr deta ha jis sy predicton asan ho jati ha
scaler = MinMaxScaler()          #yeh min max scaler es liye use krty hain Q k hamary data main kch values bht zada bari hoti hain or kch values bht zada choti hain yeh dono values ko esy scale krta ha k hamari values main gap bht zada nai rehta jis ki waja graph easy fit ho jata ha
scaler.fit(X)    #jo min max scaler ka object banaya tha us k ander X(features ki values )  enter ki
X = scaler.transform(X)    #yeh transform ka function chnage kr deta ha 0 sy 1 tk values ko X ki values ko or duara usko X k ander store krwa diya



#MAchine learning steps
from sklearn.model_selection import train_test_split          #yeh hamary liye test data or trained data ko randomly alag kr dy ga jis sy hum apny model ko check kr sakty hain 



#X hamary features thy or Y hamary labels thy jo hum ny apny data sy alag kiye thy
#testing data 0.2 ka matlab 20 percent select kiya test k liye
#random seed hota ha matlab random state =42 ka matlab k yeh jo data randomly generate kr k test or train waly main store kry ga agr next time hum chahty hain k hamy yehi data dubara mily jub yeh function lagain to huamain yehi values random_state=42 likhna pry ga
X_train , X_test , y_train , y_test = train_test_split(X , Y , test_size=0.2 , random_state=42) 





#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 25, random_state = 42)        #This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower.

# Train the model on training data
model.fit(X_train, y_train)


prediction = model.predict(X_test)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction))


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
print(cm)

#Print individual accuracy values for each class, based on the confusion matrix
print("Benign = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("Malignant = ",   cm[1,1] / (cm[0,1]+cm[1,1]))


#importances = list(model_RF.feature_importances_)
feature_imp = pd.Series(model.feature_importances_, index=features_list).sort_values(ascending=False)
print(feature_imp)