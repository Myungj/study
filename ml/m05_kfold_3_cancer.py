import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC : ", scores, "\n cross_val_score", round(np.mean(scores),4))

# ACC :  [0.87912088 0.97802198 0.92307692 0.89010989 0.9010989 ] 
#  cross_val_score 0.9143