import warnings
warnings.filterwarnings("ignore")

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# Define hyperparameter grids for tuning each model
param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [3, 5, 7, 10],
    'rf__min_samples_leaf': [1, 2, 5]
}

param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}

param_grid_svm = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# List of pipelines and parameter grids
pipelines_and_params = [
    # Random Forest pipelines
    ('rf-orig', Pipeline([('rf', RandomForestClassifier(random_state=seed))]), param_grid_rf),
    ('rf-std', Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=seed))]), param_grid_rf),
    ('rf-norm', Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier(random_state=seed))]), param_grid_rf),
    
    # KNN pipelines
    ('knn-orig', Pipeline([('knn', KNeighborsClassifier())]), param_grid_knn),
    ('knn-std', Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]), param_grid_knn),
    ('knn-norm', Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier())]), param_grid_knn),
    
    # SVM pipelines
    ('svm-orig', Pipeline([('svm', SVC())]), param_grid_svm),
    ('svm-std', Pipeline([('scaler', StandardScaler()), ('svm', SVC())]), param_grid_svm),
    ('svm-norm', Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())]), param_grid_svm),
]

# Apply GridSearchCV for each model to find the best one and collect results for the boxplot
best_model = None
best_score = 0
best_params = None
best_scaler = None

results = []  # To store accuracy results for each pipeline
names = []    # To store pipeline names for boxplot labels

for name, model, param_grid in pipelines_and_params:
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid.fit(X_train, y_train)
    
    # Append the cross-validation results to results list
    results.append(grid.cv_results_['mean_test_score'])
    names.append(name)

    print(f"{name.upper()} - Best: {grid.best_score_:.3f} using {grid.best_params_}")
    
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_scaler = grid.best_estimator_.named_steps['scaler'] if 'scaler' in grid.best_estimator_.named_steps else None

# Final model after tuning
print(f"\nBest model: {best_model}")
print(f"Best score: {best_score:.3f}")
print(f"Best parameters: {best_params}")

# Ensure a scaler is always used
if best_scaler is None:
    print("Best model did not use a scaler. Applying StandardScaler to the data.")
    best_scaler = StandardScaler()
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)
else:
    X_train_scaled = best_scaler.fit_transform(X_train)
    X_test_scaled = best_scaler.transform(X_test)

# Refit the final model on scaled data
model_key = [key for key in best_model.named_steps if 'rf' in key or 'svm' in key or 'knn' in key][0]
best_model.named_steps[model_key].fit(X_train_scaled, y_train)

# Save the best model and scaler
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(best_scaler, 'scaler.pkl')
print("Best model and scaler saved as 'best_model.pkl' and 'scaler.pkl'")

# Optionally evaluate the best model on the test set
test_score = best_model.named_steps[model_key].score(X_test_scaled, y_test)
print(f"Test set accuracy: {test_score:.3f}")

# Boxplot of the accuracy of each model
plt.figure(figsize=(12, 8))
plt.boxplot(results, labels=names, showmeans=True)
plt.title("Comparison of Model Accuracies (Original, Standardized, and Normalized Data)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()
