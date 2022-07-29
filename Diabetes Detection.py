# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Loading Dataset
data = pd.read_csv("Diabities.csv")


# Analysing the dataset
x = data.iloc[:, :-1]
y = data.iloc[:, -1]


# Splitting the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=25, random_state=0)


# Applying Logistic Regression and Evaluation
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = round(accuracy_score(y_pred, y_test), 2) * 100
print(f"Accuracy : {accuracy}")