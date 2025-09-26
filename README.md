# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.	Start with initial values for weights and bias (often zeros or small random numbers).
2.	For each iteration:
Use the current weights and bias to predict outputs for the training data.
  Calculate the error (difference) between predicted outputs and actual outputs.
Find how much the weights and bias need to change (the direction that reduces the error).
  Update the weights and bias slightly using a learning rate (small step size).
3.	Check if the error is small enough or if the maximum number of iterations is reached.
4.	If not, repeat step 2.
5.	When finished, final weights and bias represent the trained model that can be used for predictions.

```

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KARTHIK.I
RegisterNumber:  212224040121
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1)*X.T.dot(errors))
    return theta
data=pd.read_csv('/content/50_Startups (3).csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```

## Output:
<img width="693" height="767" alt="Screenshot 2025-09-23 152118" src="https://github.com/user-attachments/assets/b7414880-85f1-43ce-b921-ffba9c3aa809" />

<img width="587" height="590" alt="image" src="https://github.com/user-attachments/assets/7c106804-9e9a-4843-ad6f-835396ee8fd3" />


<img width="693" height="888" alt="Screenshot 2025-09-23 152143" src="https://github.com/user-attachments/assets/46861dd7-af2a-4b09-9e3d-e2b140a5a3c9" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
