# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:00:08 2024

@author: User
"""

import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import os
from graphviz import Source
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, mean_squared_error, mean_absolute_error, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load csv
df = pd.read_csv('filtered_ANOVA_CHI.csv')
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

# Perform SMOTE
from imblearn.over_sampling import SMOTE

# Display chart of Y after SMOTE
#y.plot.pie(autopct='%.2f')
#plt.title('Distribution of loan_status before SMOTE')
#plt.show()

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
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3, random_state=42)

# Build a model
# Create an instance of the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the Random Forest model to the training data
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Get the accuracy of the model  
# Evaluate the performance of the model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Rating:", accuracy)

# Get the F1 Score of the model
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-score:", f1)

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

# Visualize the decision tree
# Choose one of the decision trees from the Random Forest (e.g., the first tree)
tree = model.estimators_[0]

# Convert feature names to strings
feature_names = inputs.columns.astype(str)

# Convert class names to strings
class_names=[' Approved', ' Rejected']

# Export the decision tree as a DOT file
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=feature_names,  
                           class_names=class_names,  
                           filled=True, rounded=True,  
                           special_characters=True)

# Set the path to the 'dot' executable
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/dot_data.exe'

# Visualize the decision tree
graph = Source(dot_data)
graph.render("random_forest_tree.png")
graph.view()  # Display the visualization

# Classification report
print(classification_report(y_test, y_pred))
