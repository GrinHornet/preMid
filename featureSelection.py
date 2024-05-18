# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:20:05 2024

@author: User
"""

# Import Libraries
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway

# Loading dataset
dataset = pd.read_csv('cleaned_loan.csv')

def round_decimals_to_whole_numbers(df):
    """
    Detect decimal values in numerical columns and round them to the nearest whole number.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing numerical columns with decimal values.
    
    Returns:
    pd.DataFrame: The DataFrame with decimal values rounded to whole numbers.
    """
    # Select numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    # Round decimal values to the nearest whole number
    for column in numerical_columns:
        df[column] = df[column].round().astype(int)
    
    return df

df_rounded = round_decimals_to_whole_numbers(dataset)

# Drop the specified columns
columns_to_drop = ['no_of_dependents', ' education', ' self_employed']
x = dataset.drop(columns=columns_to_drop).iloc[:,:-1]
y = dataset.iloc[:,-1]

# Function to encode the categorical values to numerical values of X
def encode_dataframe(X):
    encoded_df = X.copy()  # Make a copy of the original DataFrame to avoid modifying it
    
    # Iterate through each column in the DataFrame
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    
    return encoded_df

# Encodes categorical variables in the X using LabelEncoder
new_x = encode_dataframe(x)
new_y = y
# merge X and y columns into encode_df dataFrame
encode_df = pd.concat([new_x, new_y], axis=1)

# Encode the categorical Y variable
label_encoder = LabelEncoder()
new_y_encoded = label_encoder.fit_transform(new_y)

#==============================================================================================

# Calculating ANOVA correlation between independent variables and dependent variable
# Create an empty dictionary to store the ANOVA correlation for each feature
anova_corr = {}

# Calculating ANOVA correlation between X independent variables and the Y dependent variable
for column in new_x.columns:
    groups = [new_y_encoded[new_x[column] == val] for val in np.unique(new_x[column])]
    if len(groups) > 1:  # Ensure there are at least two groups to compare
        f_stat, p_value = f_oneway(*groups)
        
        # Calculating the critical F-value
        k = len(new_x[column].unique())  # Number of groups for this specific variable
        N = len(new_x)  # Total number of observations
        df_between = k - 1  # Degrees of freedom for the model
        df_within = N - k  # Degrees of freedom for the error
        critical_f = scipy.stats.f.ppf(1 - 0.05, df_between, df_within)  # Critical F-value for α = 0.05

        remark = "Significant" if f_stat > critical_f else "Not Significant"
        anova_corr[column] = {'f_stat': f_stat, 'p_value': p_value, 'critical_f': critical_f, 'remark': remark}

anova_corr_df = pd.DataFrame(anova_corr).T
anova_corr_df['f_stat'] = anova_corr_df['f_stat'].astype(float)
anova_corr_df['p_value'] = anova_corr_df['p_value'].astype(float)
anova_corr_df['critical_f'] = anova_corr_df['critical_f'].astype(float)
anova_corr_df['remark'] = anova_corr_df['remark'].astype(str)
print(anova_corr_df)
# Split columns that are significant and not significant 
not_significant_columns = anova_corr_df[anova_corr_df['remark'] == 'Not Significant'].index.tolist()
significant_columns = anova_corr_df[anova_corr_df['remark'] == 'Significant'].index.tolist()

# Drop not significant columns from the dataset
new_x = new_x.drop(columns=not_significant_columns)

# To export x with the categorical x
export_x_df = dataset.drop(columns=not_significant_columns)

# Export the resulting new_x and y to a new CSV file (After Chi square export)
#result_df = pd.concat([new_x, y], axis=1)
#result_df.to_csv('filtered_loan.csv', index=False)

# Exporting new CSV before Chi-square
export_x_df.to_csv('filtered_loan.csv', index=False)

#============================================================================================== 

print("ANOVA Correlation (x to y):")
print(anova_corr_df)
print()
print("Degrees of Freedom (Between):", df_between)
print("Degrees of Freedom (Within):", df_within)


