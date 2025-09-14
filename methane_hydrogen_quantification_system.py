# Data analysis and manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer


# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score

# Model selection and evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# Radial Bias Function SVM
from sklearn.svm import SVR

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Artificial Neural Network
import tensorflow as tf
from tensorflow import keras

# Import Dataset
df = pd.read_csv('/content/Hydro_Methane.csv')

df.head(10)

df.columns

# Number of features and Samples (Columns and Rows)
df.shape

df.info()

# Duplicate data
duplicates = df.duplicated()

duplicates.sum()

df.isnull().sum()

# Dropping uncessary features
wavelengths = df.drop(['ID', 'Ar_flow', 'ArH_flow', 'Meth_flow', 'H_ppm', 'Meth_ppm'],axis='columns')

#'H_ppm'
H_wavelengths = df.drop(['ID', 'Ar_flow', 'ArH_flow', 'Meth_flow', 'Meth_ppm'],axis='columns')

#'Meth_ppm'
M_wavelengths = df.drop(['ID', 'Ar_flow', 'ArH_flow', 'Meth_flow', 'H_ppm'],axis='columns')

x = np.arange(195,1105, 0.2)

plt.plot(x, wavelengths.iloc[77], c='red', alpha=0.50)

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Hydrogen PPM ' + str(np.round(H_wavelengths.iloc[77]['H_ppm'])))
plt.legend()
plt.show()

plt.plot(x, wavelengths.iloc[77], c='red', alpha=0.50)
plt.plot(x, wavelengths.iloc[1000], c='blue', alpha=0.50)

plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Hydrogen PPM ' + str(np.round(H_wavelengths.iloc[77]['H_ppm'])) + ' & ' + str(np.round(H_wavelengths.iloc[1000]['H_ppm'])))
plt.legend()
plt.show()

X = H_wavelengths.iloc[:,1:]
y = H_wavelengths['H_ppm']

feature_names = X.columns.tolist()

X_valid, X_train, y_valid, y_train = train_test_split(X, y, test_size=0.3, random_state=50)

# Numerical Columns
numerical_columns = X_train.select_dtypes(include=['float','int']).columns

# The transformer for numerical features
numerical_transformer = Pipeline(steps=[
    ('Standard Scaler', StandardScaler())
])

# Create Random Forest regressor model
rf_regressor = RandomForestRegressor()

# Fitting model to training data
rf_regressor.fit(X_train, y_train)

# feature importances
feature_importances = rf_regressor.feature_importances_

# Get indices of top 50 features based on importance
top_50_indices = np.argsort(feature_importances)[::-1][:50]

# Get the names and importances of top 50 features
top_50_features = [feature_names[i] for i in top_50_indices]
top_50_importances = [feature_importances[i] for i in top_50_indices]

plt.figure(figsize=(10, 6))
plt.bar(range(len(top_50_features)), top_50_importances, align='center', color='skyblue')
plt.xticks(range(len(top_50_features)), top_50_features, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 50 Feature')
plt.tight_layout()
plt.show()

# Calculate the mean of feature importance
mean_importance = np.mean(feature_importances)

# Select features with importance above the mean
selected_features = [numerical_columns[i] for i, importance in enumerate(feature_importances) if importance > mean_importance]

# Sort feature importances in descending order and get indices of top 50 features
top_50_indices = np.argsort(feature_importances)[::-1][:50]

# Select top 50 features from numerical_columns based on their indices
top_50_features = [numerical_columns[i] for i in top_50_indices]

print(top_50_features)

preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, top_50_features),
])

preprocessor

# List of scoring metrics
metrics = {
    'MAE': make_scorer(mean_absolute_error),
    'MSE': make_scorer(mean_squared_error),
    'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
    'R2': 'r2'
}

def best_hyperparameters(grid_search_results, metrics):

    for metric in metrics:
        test_metric = "split0_test_" + metric

        if test_metric not in grid_search_results:
            print("Scoring metric {} not found in grid search results.".format(metric))
            continue

        if metric == 'R2':
            best_index = np.argmax(grid_search_results[test_metric])
        else:
            best_index = np.argmin(grid_search_results[test_metric])

        best_params = grid_search_results['params'][best_index]
        print("Best hyperparameters for {}: {}".format(metric, best_params))

"""SVR (RBF)"""

RBF_SVR_model = SVR(kernel='rbf')

operations = [("preprocessor", preprocessor), ("regressor", RBF_SVR_model)]

RBF_SVR_model_pipeline = Pipeline(steps=operations)

RBF_SVR_parameters = {
    'regressor__C': [0.1, 1, 10, 100],  # Regularization parameter
    'regressor__gamma': [0.01, 0.1, 1, 10],  # Kernel coefficient
    'regressor__epsilon': [0.1, 0.2, 0.5]  # Epsilon parameter
}

grid_search_RBF_SVR_model = GridSearchCV(RBF_SVR_model_pipeline, RBF_SVR_parameters, cv=5, scoring=metrics, refit=False)

grid_search_RBF_SVR_model.fit(X_train, y_train)

pd.DataFrame(grid_search_RBF_SVR_model.cv_results_).head()

best_hyperparameters(grid_search_RBF_SVR_model.cv_results_, metrics)

RBF_SVR_model = SVR(kernel='rbf', C=0.1, epsilon=0.1, gamma=10)

operations = [("preprocessor", preprocessor), ("regressor", RBF_SVR_model)]
RBF_SVR_model_pipeline = Pipeline(steps=operations)

RBF_SVR_model_results = cross_validate(RBF_SVR_model_pipeline, X_train, y_train, scoring = metrics, cv=5, return_train_score = True)

pd.DataFrame(RBF_SVR_model_results)

"""Random Forest"""

RF_regressor_model = RandomForestRegressor()

operations = [("preprocessor", preprocessor), ("regressor", RF_regressor_model)]

RF_regressor_pipeline = Pipeline(steps=operations)

RF_regressor_parameters = {
    'regressor__n_estimators': [50, 100, 200],                # Number of trees in the forest
    'regressor__max_depth': [None, 10, 20],                   # Maximum depth of the trees
    'regressor__min_samples_split': [2, 5, 10],               # Minimum number of samples required to split an internal node
    'regressor__min_samples_leaf': [1, 2, 4],                 # Minimum number of samples required to be at a leaf node
    'regressor__max_features': [None, 'sqrt', 0.5]            # Number of features to consider when looking for the best split
}

grid_search_RF_regressor_model = GridSearchCV(RF_regressor_pipeline, RF_regressor_parameters, cv=5, scoring=metrics, refit=False)

grid_search_RF_regressor_model.fit(X_train, y_train)

pd.DataFrame(grid_search_RF_regressor_model.cv_results_).head()

best_hyperparameters(grid_search_RF_regressor_model.cv_results_, metrics)

RF_regressor_model = RandomForestRegressor(
    max_depth=10,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100
)

operations = [("preprocessor", preprocessor), ("regressor", RF_regressor_model)]
RF_regressor_pipeline = Pipeline(steps=operations)

RF_model_results = cross_validate(RF_regressor_pipeline, X_train, y_train, scoring = metrics, cv=5, return_train_score = True)

pd.DataFrame(RF_model_results)

"""Artificial Neural Network"""

from scikeras.wrappers import KerasClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

# Define the function to create the ANN model
def create_ann_model(epochs=10, batch_size=32, dropout_rate=0.1):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create KerasRegressor wrapper for the ANN model
ANN_model = KerasRegressor(build_fn=create_ann_model, dropout_rate=0.1, verbose=0)

# Define hyperparameters for grid search
param_grid = {
    'regressor__epochs': [10, 20, 30],        # Number of training epochs
    'regressor__batch_size': [32, 64, 128],   # Batch size for training
    'regressor__optimizer': ['adam', 'sgd', 'rmsprop']  # Optimizer algorithm
}



# Create a pipeline with preprocessing and the ANN model
operations = [("preprocessor", StandardScaler()), ("regressor", ANN_model)]
ANN_model_pipeline = Pipeline(steps=operations)

# Perform grid search
grid_search_ANN_model = GridSearchCV(ANN_model_pipeline, param_grid=param_grid, cv=5, scoring=metrics, refit=False)
grid_search_ANN_model.fit(X_train, y_train)

pd.DataFrame(grid_search_ANN_model.cv_results_).head()

best_hyperparameters(grid_search_ANN_model.cv_results_, metrics)

# ANN model
def create_ann_model(epochs=30, batch_size=32, optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    # Compile the model with the specified optimizer
    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    return model

# Create KerasRegressor wrapper for the ANN model
ANN_model = KerasRegressor(build_fn=create_ann_model, batch_size=32, epochs=30, optimizer='adam', verbose=0)

# Create a pipeline with preprocessing and the ANN model
operations = [("preprocessor", StandardScaler()), ("regressor", ANN_model)]
ANN_model_pipeline = Pipeline(steps=operations)

# Fit the pipeline on the training data
ANN_model_pipeline.fit(X_train, y_train)

ANN_model_results = cross_validate(ANN_model_pipeline, X_train, y_train, scoring = metrics, cv=5, return_train_score = True)

pd.DataFrame(ANN_model_results)

"""Model Comparison"""

def calculate_metrics_mean(results):
    # Extract the relevant columns from the results data
    test_MAE = results['test_MAE']
    train_MAE = results['train_MAE']
    test_MSE = results['test_MSE']
    train_MSE = results['train_MSE']
    test_RMSE = results['test_RMSE']
    train_RMSE = results['train_RMSE']
    test_R2 = results['test_R2']
    train_R2 = results['train_R2']

    # Calculate the mean of each metric
    mean_test_MAE = np.mean(test_MAE)
    mean_train_MAE = np.mean(train_MAE)
    mean_test_MSE = np.mean(test_MSE)
    mean_train_MSE = np.mean(train_MSE)
    mean_test_RMSE = np.mean(test_RMSE)
    mean_train_RMSE = np.mean(train_RMSE)
    mean_test_R2 = np.mean(test_R2)
    mean_train_R2 = np.mean(train_R2)

    # Return the mean values
    return {
        'mean_est_MAE': mean_test_MAE,
        'mean_train_MAE': mean_train_MAE,
        'mean_test_MSE': mean_test_MSE,
        'mean_train_MSE': mean_train_MSE,
        'mean_test_RMSE': mean_test_RMSE,
        'mean_train_RMSE': mean_train_RMSE,
        'mean_test_R2': mean_test_R2,
        'mean_train_R2': mean_train_R2
    }

calculate_metrics_mean(RBF_SVR_model_results)

calculate_metrics_mean(RF_model_results)

calculate_metrics_mean(ANN_model_results)

model_labels = ['SVR RBF', 'RF', 'ANN']

MAE_results = [RBF_SVR_model_results['test_MAE'], RF_model_results['test_MAE'] ,ANN_model_results['test_MAE']]

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=MAE_results)
plt.title('Testing MAE')
plt.xlabel('Models')
plt.ylabel('Testing MAE')
plt.xticks(ticks=np.arange(len(model_labels)), labels=model_labels)
plt.show()

MSE_results = [RBF_SVR_model_results['test_MSE'], RF_model_results['test_MSE'] ,ANN_model_results['test_MSE']]

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=MSE_results)
plt.title('Testing MSE')
plt.xlabel('Models')
plt.ylabel('Testing MSE')
plt.xticks(ticks=np.arange(len(model_labels)), labels=model_labels)
plt.show()

RMSE_results = [RBF_SVR_model_results['test_RMSE'], RF_model_results['test_RMSE'] ,ANN_model_results['test_RMSE']]

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=RMSE_results)
plt.title('Testing RMSE')
plt.xlabel('Models')
plt.ylabel('Testing RMSE')
plt.xticks(ticks=np.arange(len(model_labels)), labels=model_labels)
plt.show()

R2_results = [RBF_SVR_model_results['test_R2'], RF_model_results['test_R2'] ,ANN_model_results['test_R2']]

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=R2_results)
plt.title('Testing R2')
plt.xlabel('Models')
plt.ylabel('Testing R2')
plt.xticks(ticks=np.arange(len(model_labels)), labels=model_labels)
plt.show()

X = H_wavelengths.iloc[:,1:]
y = H_wavelengths['H_ppm']

# Final Model

df = pd.read_csv('/content/Hydro_Methane.csv')

#'H_ppm'
H_wavelengths = df.drop(['ID', 'Ar_flow', 'ArH_flow', 'Meth_flow', 'Meth_ppm'],axis='columns')

X = H_wavelengths.iloc[:,1:]
y = H_wavelengths['H_ppm']

#'H_ppm'
H_wavelengths = df.drop(['ID', 'Ar_flow', 'ArH_flow', 'Meth_flow', 'Meth_ppm'],axis='columns')

# The transformer for numerical features
numerical_transformer = Pipeline(steps=[
    ('Standard Scaler', StandardScaler())
])

# Select top 50 features from numerical_columns based on their indices
top_50_features = ['678', '261.4', '430.8', '261.2', '299.6', '272', '386', '677.8', '305', '776', '273.8', '300.2', '654.8', '775.8', '270.6', '305.2', '272.2', '386.2', '273.6', '305.4', '386.6', '260', '292.2', '320.6', '274', '320.2', '273', '272.8', '259.8', '386.4', '287.4', '332.8', '257.2', '258', '300.4', '302.6', '320.4', '269.4', '273.4', '272.6', '262', '257', '274.2', '291.8', '299.8', '307.6', '669.2', '678.2', '421.6', '385.8']

preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, top_50_features),
])

preprocessor

RF_regressor_model = RandomForestRegressor(
    max_depth=10,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100
)

operations = [("preprocessor", preprocessor), ("regressor", RF_regressor_model)]
RF_regressor_pipeline = Pipeline(steps=operations)