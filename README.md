# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Employee.csv dataset and display the first few rows.
2. Check dataset structure and find any missing values.
3. Display the count of employees who left vs stayed.
4. Encode the "salary" column using LabelEncoder to convert it into numeric values.
5. Define features x with selected columns and target y as the "left" column.
6. Split the data into training and testing sets (80% train, 20% test).
7. Create and train a DecisionTreeClassifier model using the training data.
8. Predict the target values using the test data.
9. Evaluate the model’s accuracy using accuracy score.
10. Predict whether a new employee with specific features will leave or not.


## Program:
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Jayaseelan U
RegisterNumber:  212223220039
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
![1](https://github.com/user-attachments/assets/d58659c9-58f2-4a0b-8262-b8c4c31cffad)
![2](https://github.com/user-attachments/assets/bf105051-e9bf-4971-86a2-a2a5dcd0127c)
![3](https://github.com/user-attachments/assets/93af8f87-b2f2-4f9f-9276-0db8f80ec105)






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
