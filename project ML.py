import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
from sklearn.metrics import accuracy_score

# loading the data from csv file to a Pandas DataFrame
medical_dataset = pd.read_csv('insurance.csv')

medical_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)
medical_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)
medical_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


X = medical_dataset.drop(columns='charges')
Y = medical_dataset['charges']


# ????? ?????? ?????? ????? ?????? ??? ?????? ????? ????????? ???? ????
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

print("=============================================================")


# loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train, Y_train)


# prediction on training data
training_data_prediction =regressor.predict(X_train)

print("=============================================================")

test_data_prediction =regressor.predict(X_test)

# R squared value
score_test1 = metrics.r2_score( Y_test ,test_data_prediction )
print('Accurecy for test about LinearRegression : ', score_test1) ##accuracy

print("=============================================================")


#RandomForestClassifier Model

#Applying RandomForestRegressor Model


#better performance but makes your code slower. You should choose as high value as your processor can
#handle because this makes your predictions stronger and more stable
RandomForestClassifierModel = RandomForestRegressor(n_estimators=100 , random_state=44)
RandomForestClassifierModel.fit(X_train, Y_train)


#calculating Details
score_test2=RandomForestClassifierModel.score(X_test, Y_test)
print( " Accurecy for test about RandomForestRegressor" , score_test2)



print("=============================================================")


from sklearn.tree import DecisionTreeRegressor



model_Decision_Tree=DecisionTreeRegressor(random_state=44 )
model_Decision_Tree.fit(X_train, Y_train)

score_test3=model_Decision_Tree.score(X_test,Y_test)
y_pred1=model_Decision_Tree.predict(X_test)

# MSE_Decision_Tree=np.sqrt(mean_squared_error(y_test,y_pred))
print("Accurecy for test about Decision_Tree = ",score_test3)

print("=============================================================")

if score_test1>score_test2 and score_test1>score_test3 :
    print("the best model in this project is : Linear Regression")

elif score_test2>score_test1 and score_test2>score_test3:
    print("the best model in this project is : random forest model ")

else: 
    print("the best model in this project is : Decision_Tree ")


print("=============================================================")

