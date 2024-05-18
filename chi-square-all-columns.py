# -*- coding: utf-8 -*-
"""
Created on Fri May 17 07:36:37 2024

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# Load CSV
df = pd.read_csv('cleaned_loan.csv')

def convert_to_categories(data, column, bins, labels):
    return pd.cut(data[column], bins=bins, labels=labels, right=False)

# Define income_annum categories
income_bins = [600000, 2066667, 3533334, 5000001, 6466668, 7933335, 9400001]
loan_bins = [1800000, 6650000, 11500000, 16350000, 21200000, 26050000, 30900001]
term_bins = [2, 5, 8, 11, 14, 17, 21]
cibil_bins = [330, 420, 510, 600, 690, 780, 901]
cav_bins = [200000, 2483333, 4766666, 7049999, 9333332, 11616665, 13900001]
rav_bins = [300000, 3783333, 7266666, 10750000, 14233333, 17716666, 21200001]
lav_bins = [1900000, 6800000, 11700000, 16600000, 21500000, 26400000, 31300001]
bav_bins = [500000, 2266666, 3983333, 5700000, 7416666, 9133333, 11100001]

# Convert Labels to Integers
labels = [1, 2, 3, 4, 5, 6]

# Define the columns to be converted
columns_to_convert = {
    ' income_annum': income_bins,
    ' loan_amount': loan_bins,
    ' loan_term': term_bins,
    ' cibil_score': cibil_bins,
    ' residential_assets_value': rav_bins,
    ' commercial_assets_value': cav_bins,
    ' luxury_assets_value': lav_bins,
    ' bank_asset_value': bav_bins
}

# Convert numerical columns to categorical ranges
for column, bins in columns_to_convert.items():
    df[column] = convert_to_categories(df, column, bins, labels)

# Separate X and y
X = df.drop([' loan_status'], axis=1)
Y = df[' loan_status']

encoded_X = (X)
encoded_Y = (Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(encoded_X, encoded_Y, test_size=0.3, random_state=42)

# Perform chi-square test for each feature
feature_significance = {}
for col in X_train.columns:
    contingency_table = pd.crosstab(X_train[col], Y_train)
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    feature_significance[col] = {"Significance": "Significant" if p_val < 0.05 else "Not Significant",
                                  "p-value": p_val}

# Print significance and p-values of each feature
print("\nFeature Significance and p-values:")
for feature, values in feature_significance.items():
    print(f"{feature}: {values['Significance']}, p-value: {values['p-value']}")

# Create DataFrame to store chi-square test results
chi2_df = pd.DataFrame(feature_significance).T
chi2_df.index.name = 'Feature'
chi2_df.reset_index(inplace=True)

# Filter the original DataFrame based on significant features
significant_X_columns = chi2_df[chi2_df['Significance'] == 'Not Significant']['Feature'].tolist()
filtered_df = df.drop(significant_X_columns, axis=1)

# Calculate correlation matrix for predictors (independent variables)
predictors_corr = X_train.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(predictors_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Predictors')
plt.show()


# Export the filtered DataFrame to CSV
filtered_df.to_csv('chi_square_filtered_loan.csv', index=False)

# Display chi2_df in the variable explorer
chi2_df
