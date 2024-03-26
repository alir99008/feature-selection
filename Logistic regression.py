# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:50:37 2022

@author: Hp
"""

#LOGISTIC REGRESSION hum es liye use krty hain Q k yeh hamari sari values ko sygmoid function k zariye solve krta ha jis ki values 0-1 k beach main krta ha  or yeh loss ko B kam sy kam krta ha


# https://youtu.be/WUqBG-hW_f4

"""
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
## 'data.frame':    569 obs. of  31 variables:
##  $ diagnosis              : Factor w/ 2 levels "Benign","Malignant": 2 2 2 2 2 2 2 2 2 2 ...
##  $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
##  $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
##  $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
##  $ area_mean              : num  1001 1326 1203 386 1297 ...
##  $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
##  $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
##  $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
##  $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
##  $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
##  $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
##  $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
##  $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
##  $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
##  $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
##  $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
##  $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
##  $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
##  $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
##  $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
##  $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
##  $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
##  $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
##  $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
##  $ area_worst             : num  2019 1956 1709 568 1575 ...
##  $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
##  $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
##  $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
##  $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
##  $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
##  $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...


"""

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("breast cancer dataset.csv")

print(df.describe().T)  #Values need to be normalized before fitting. 


print(df.isnull().sum())
#df = df.dropna()

#Rename Dataset to Label to make it easy to understand
df = df.rename(columns={'diagnosis':'Label'})     #hum apny diagnosis waly column ko as a Label rakh k deal krain gy gy es liye hm eska name change kr k label rakh dain gyy
print(df.dtypes)

#Understand the data 
sns.countplot(x="Label", data=df) #M - malignant   B - benign       #yeh hamaain simple graph plot kr k dy dy ga bar graph

sns.distplot(df['radius_mean'], kde=False)

print("correlation = ",df.corr())                #yeh hamain har column ka corelation find kr k dy ga hamary  data main jitny B column howy unka subka

corrMatrix = df.corr()
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
#sns.heatmap(df.iloc[:, 1:6:], annot=True, linewidths=.5, ax=ax)
sns.heatmap(corrMatrix, annot=False, linewidths=.5, ax=ax)          #yeh graph pehly find howy correlation ko dekhty 1 graph plot kr dy ga jo k bht zabardast define krta ha


#Replace categorical values with numbers
df['Label'].value_counts()                          #yeh pehly B or M ko btay ga k total kitni bar aay ga

categories = {"B":1, "M":2}
df['Label'] = df['Label'].replace(categories)         #Hum B or M ko sirf es liye change krain gy Q k hamara data Sirf Numeric values pr he kam krta ha JAB k B or M Char hain...


#Define the dependent variable that needs to be predicted (labels)
#Labels
Y = df["Label"].values                   #es main hum ny label ki values ko Y k ander store krwaya jo k hum apny model k as a label dain gyy

#Features
X=df.drop(labels=["Label" , "id"] , axis=1)      #es main hum ny label wala column es liye drop kiya Q k wo hum as a label use kr rhy thy es liye usko features main use nai kr sakty thy  or id wala column es liye drop kiya Q k uski koi zarorat nai thi


#Without scaling the error would be large. Near 100% for no disease class. 
from sklearn.preprocessing import MinMaxScaler
#Min Max Scaler hamari bht bari or choti values ko 0 sy 1 k beach main change kr deta ha jis sy predicton asan ho jati ha
scaler = MinMaxScaler()          #yeh min max scaler es liye use krty hain Q k hamary data main kch values bht zada bari hoti hain or kch values bht zada choti hain yeh dono values ko esy scale krta ha k hamari values main gap bht zada nai rehta jis ki waja graph easy fit ho jata ha
scaler.fit(X)    #jo min max scaler ka object banaya tha us k ander X(features ki values )  enter ki
X = scaler.transform(X)    #yeh transform ka function chnage kr deta ha 0 sy 1 tk values ko X ki values ko or duara usko X k ander store krwa diya

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split                    #yeh hamary liye test data or trained data ko randomly alag kr dy ga jis sy hum apny model ko check kr sakty hain 


#X hamary features thy or Y hamary labels thy jo hum ny apny data sy alag kiye thy
#testing data 0.2 ka matlab 20 percent select kiya test k liye
#random seed hota ha matlab random state =42 ka matlab k yeh jo data randomly generate kr k test or train waly main store kry ga agr next time hum chahty hain k hamy yehi data dubara mily jub yeh function lagain to huamain yehi values random_state=42 likhna pry ga

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Fir the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


prediction = model.predict(X_test)                        #Jab hum ny apna model bana liya us k bad hum ny thora test data jo 20% tha jisko spliter k zariye alag kiya tha wo apny model k dy kr check krain gy k result thk aarhy hain k nai

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction))    #jo prediction aai usko ko jo hamary labels thy us sy chcek krain gy k kitna percent correct ha



#Confusion Matrix
from sklearn.metrics import confusion_matrix                           #confusion metrix ka matlab k yeh btata ha k hamary kitny labaels saii predct kye hamary model ny or kitny labels galat predict kiye
cm = confusion_matrix(y_test, prediction)                   #[[70  1]         #Total 72 B thy jin main sy 70 saii predict kiye or 2 galat  or 42 M thy jin main sy 41 saii predict kiye or 1 galat
                                                            #[ 2 41]]
print(cm)

#Print individual accuracy values for each class, based on the confusion matrix
print("With Lung disease = ", cm[0,0] / (cm[0,0]+cm[1,0]))
print("No disease = ",   cm[1,1] / (cm[0,1]+cm[1,1]))


