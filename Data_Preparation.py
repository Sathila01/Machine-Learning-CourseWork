import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv('spambase/spambase.data', header=None)

print(data.head())

# Split target variable from features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN Model
# Now, we will implement the KNN model using the scikit-learn library.
# We will use the Euclidean distance metric and 5 neighbors for classification:

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Fit the classifier to the data
knn.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = knn.predict(X_test)

# Evaluate KNN model using accuracy, precision, recall, and F1 score:
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_score_knn = f1_score(y_test, y_pred_knn)

print("KNN Model Evaluation:")
print("Accuracy: {:.2f}".format(accuracy_knn))
print("Precision: {:.2f}".format(precision_knn))
print("Recall: {:.2f}".format(recall_knn))
print("F1 Score: {:.2f}".format(f1_score_knn))

# Create Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the data
dt.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt.predict(X_test)

# Evaluate Decision Tree model using accuracy, precision, recall, and F1 score:
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_score_dt = f1_score(y_test, y_pred_dt)

print("Decision Tree Model Evaluation:")
print("Accuracy: {:.2f}".format(accuracy_dt))
print("Precision: {:.2f}".format(precision_dt))
print("Recall: {:.2f}".format(recall_dt))
print("F1 Score: {:.2f}".format(f1_score_dt))

