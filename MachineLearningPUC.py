
import warnings
warnings.filterwarnings("ignore")

# Imports necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dataset = pd.read_csv(url, delimiter=";")

# Step 2: Show the dataset head
print(dataset.head())

# Preprocessing
X = dataset.iloc[:, :-1].values  # All columns except the last
y = dataset.iloc[:, -1].values    # Last column (quality)

# Separate into training and testing sets
test_size = 0.20
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)

# Parameters and partitions for cross-validation
scoring = 'accuracy'
num_partitions = 10
kfold = StratifiedKFold(n_splits=num_partitions, shuffle=True, random_state=seed)

# List of models including Random Forest for better performance
models = [
    ('RF', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed)),
    ('KNN', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

# Evaluation of models
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot comparison of models
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Comparison of Models')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Pipelines for different treatments on the data
pipelines = [
    ('RF-orig', Pipeline([('rf', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed))])),
    ('KNN-orig', Pipeline([('knn', KNeighborsClassifier())])),
    ('NB-orig', Pipeline([('nb', GaussianNB())])),
    ('SVM-orig', Pipeline([('svm', SVC())])),
    ('RF-padr', Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed))])),
    ('KNN-padr', Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])),
    ('NB-padr', Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())])),
    ('SVM-padr', Pipeline([('scaler', StandardScaler()), ('svm', SVC())])),
    ('RF-norm', Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed))])),
    ('KNN-norm', Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier())])),
    ('NB-norm', Pipeline([('scaler', MinMaxScaler()), ('nb', GaussianNB())])),
    ('SVM-norm', Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])),
]

# Evaluation of pipelines
results = []
names = []

for name, model in pipelines:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Boxplot comparison of pipelines
fig = plt.figure(figsize=(25, 6))
fig.suptitle('Comparison of Models - Original, Standardized, and Normalized Dataset')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, rotation=90)
plt.show()

# Tuning of Random Forest
np.random.seed(7)

pipelines = [
    ('rf-orig', Pipeline([('rf', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed))])),
    ('rf-padr', Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed))])),
    ('rf-norm', Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=seed))])),
]

param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 7, 10],
    'rf__min_samples_leaf': [1, 2, 5],
}

# GridSearchCV for Random Forest
best_model = None
best_score = 0
best_scaler = None
for name, model in pipelines:
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid.fit(X_train, y_train)
    print("Without missing treatment: %s - Best: %f using %s" % (name, grid.best_score_, grid.best_params_))
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_scaler = grid.best_estimator_.named_steps['scaler'] if 'scaler' in grid.best_estimator_.named_steps else None

# Ensure that a scaler is added if no scaler was found
if best_scaler is None:
    best_scaler = StandardScaler()
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
    best_model.fit(X_train_scaled, y_train)

# Save the best model and scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(best_scaler, 'scaler.pkl')
print("Best model and scaler saved as 'best_model.pkl' and 'scaler.pkl'")
