import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)

#explore data
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))

# Outcome is the variable that tells if an observation has diabeter yes (1) or no (0)
print(diabetes.groupby('Outcome').size())

import seaborn as sns
sns.countplot(diabetes['Outcome'],label="Count")

diabetes.info()

#First, Let’s investigate whether we can confirm the connection between model complexity and accuracy for KNN model:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')

#the plot shows ideal number of neighbors is 9. 
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

diabetes_features = [x for i,x in enumerate(diabetes.columns) if i!=8]
plt.figure(figsize=(8,6))
plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)
plt.hlines(0, 0, diabetes.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('log_coef')

#Decision Tree: Accuracy on Training set is 1 and test set 0.714 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on Decision Tree training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Decision Tree test set: {:.3f}".format(tree.score(X_test, y_test)))

#Accuracy on training set is 1 that means model is overfitting, so you prune the tree to max_depth = 3
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on Decision Tree with max_depth=3 training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on Decision Tree with max_depth=3 test set: {:.3f}".format(tree.score(X_test, y_test)))

print("Feature importances:\n{}".format(tree.feature_importances_))

#Visualize Feature Importance
def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
plot_feature_importances_diabetes(tree)
plt.savefig('feature_importance')

# Random Forest 78.6 % accuracy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on Random Forest training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on Random Forest test set: {:.3f}".format(rf.score(X_test, y_test)))

#Adjust the max_features to see if it improves accuracy. It doesn't, so stick to Random Forest.
rf1 = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1.fit(X_train, y_train)
print("Accuracy on Random Forest with max_depth = 3 training set: {:.3f}".format(rf1.score(X_train, y_train)))
print("Accuracy on Random Forest with max_depth = 3 test set: {:.3f}".format(rf1.score(X_test, y_test)))

plot_feature_importances_diabetes(rf)