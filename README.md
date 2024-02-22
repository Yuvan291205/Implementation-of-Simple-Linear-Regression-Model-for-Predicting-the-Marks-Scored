# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yuvan M
RegisterNumber:  212223240188
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![Screenshot 2024-02-22 033929](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/55df46b6-5e69-4f43-87d4-6763d02d0836)
![Screenshot 2024-02-22 033938](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/24eb2c03-dcc8-4227-ab27-716e48993cb9)
![Screenshot 2024-02-22 034009](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/6441950d-81bb-4d9b-a0a3-a10bfa0d3a8c)
![Screenshot 2024-02-22 034020](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/702e3ab8-7a96-4e06-84de-7791e8872fdd)
![Screenshot 2024-02-22 034033](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/332b9e09-7985-4233-b4d3-99fb0a49b7b9)
![Screenshot 2024-02-22 034057](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/a91645df-82a1-4ef6-88e1-8179025cb726)
![Screenshot 2024-02-22 034109](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/3ff3f44c-b913-4af0-bc31-29ba070d0c02)
![Screenshot 2024-02-22 034119](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/af4c3c49-90d9-405b-a548-0898d87e0156)
![Screenshot 2024-02-22 034131](https://github.com/Yuvan291205/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849170/26ca1d4c-1d4f-4a38-972e-c9af58a5862c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
