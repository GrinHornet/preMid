# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:35:49 2024

@author: User
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

# Load csv
df = pd.read_csv('filtered_loan.csv')
df.head()

# Checking data imbalances
# Split to X(Predictors) and Y(Target)
x = df.drop([' loan_status'], axis=1)
y = df[' loan_status']

# Count instances for every category
y.value_counts()

# Display chart of Y
y.value_counts().plot.pie(autopct='%.2f')

# Check unique values in target variable
print(df[' loan_status'].unique())

# Map non-numeric values to numerical equivalents directly in the dataframe
#df[' loan_status'] = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})

# Separate X and y
x = df.drop([' loan_status'], axis=1)
y = df[' loan_status']

# Perform data balancing using over-sampling (SMOTE)
# Convert categorical to numeric values
from sklearn.preprocessing import LabelEncoder
# Now, apply LabelEncoder to any remaining categorical columns in x
label_encoders = {}
for col in x.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    x[col] = label_encoders[col].fit_transform(x[col])

# Perform SMOTE for data balancing
smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(x, y)
x_smote = pd.DataFrame(x_smote).round().astype(int)

# Count instances for every category after SMOTE
y_smote_value_counts = pd.Series(y_smote).value_counts()

# Display chart of Y after SMOTE
y_smote_value_counts.plot.pie(autopct='%.2f')
plt.title('Distribution of loan_status after SMOTE')
plt.show()
# Transfer the x_smote and y_smote to inputs and targets
inputs = x_smote
target = y_smote

def convert_to_categories(data, column, bins, labels):
    return pd.cut(data[column], bins=bins, labels=labels, right=False)

# Define income_annum categories
income_bins = [600000, 2066667, 3533334, 5000001, 6466668, 7933335, 9400001]
#loan_bins = [1800000, 6650000, 11500000, 16350000, 21200000, 26050000, 30900001]
term_bins = [2, 5, 8, 11, 14, 17, 21]
cibil_bins = [330, 420, 510, 600, 690, 780, 901]
#cav_bins = [200000, 2483333, 4766666, 7049999, 9333332, 11616665, 13900001]
#rav_bins = [300000, 3783333, 7266666, 10750000, 14233333, 17716666, 21200001]
#lav_bins = [1900000, 6800000, 11700000, 16600000, 21500000, 26400000, 31300001]
#bav_bins = [500000, 2266666, 3983333, 5700000, 7416666, 9133333, 11100001]

# Convert Labels to Integers
labels = [1, 2, 3, 4, 5, 6]

# Define the columns to be converted
columns_to_convert = {
    ' income_annum': income_bins,
    #' loan_amount': loan_bins,
    ' loan_term': term_bins,
    ' cibil_score': cibil_bins,
    #' residential_assets_value': rav_bins,
    #' commercial_assets_value': cav_bins,
    #' luxury_assets_value': lav_bins,
    #' bank_asset_value': bav_bins
}

# Convert numerical columns to categorical ranges
for column, bins in columns_to_convert.items():
    df[column] = convert_to_categories(df, column, bins, labels)

model = SVC()

# Train the model
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Get the accuracy of the model  
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Rating:", accuracy)

# Get the F1 Score of the model
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Get the precision score
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision Score:", precision)

# Calculate the Mean Squares Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate the Mean Absolute error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Show the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Train the model with only 'GENDER' and 'AGE' features
X_train_subset = X_train[['no_of_dependents', ' education']]
model.fit(X_train_subset, y_train)

##print(X_train.columns)
#print(X_train.dtypes)
# Convert data type of "Age" to numeric
X_train['no_of_dependents'] = pd.to_numeric(X_train['no_of_dependents'])
X_train[' education'] = pd.to_numeric(X_train[' education'])

# Visualization of the decision boundary
def plot_decision_boundary(X, y, model):
    # Plot decision boundary using all features
    h = .03  # step size in the mesh
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

# Plot decision boundary using two features
plt.figure(figsize=(10, 6))
plot_decision_boundary(X_train, y_train, model)
plt.title("Decision Boundary of Support Vector Machine")
plt.xlabel("no_of_dependents")
plt.ylabel(" education")
plt.show()