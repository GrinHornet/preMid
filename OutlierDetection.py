# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:20:05 2024

@author: User
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('C:\\Users\\User\\Desktop\\FOREST\\loan.csv')

# Explore the dataset to understand the structure
print(df.head())
print(df.info())
print(df.describe())

#Finding outliers and replacing it to nan value--------------------------------------------

# Define a function to calculate lower and upper limits for outliers based on percentiles
def calculate_outlier_limits(df, columns, lower_percentile=0.05, upper_percentile=0.95):
    outlier_limits = {}
    for column in columns:
        lower_limit = df[column].quantile(lower_percentile)
        upper_limit = df[column].quantile(upper_percentile)
        outlier_limits[column] = (lower_limit, upper_limit)
    return outlier_limits

# Specify the numerical columns for which to calculate outlier limits
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Plot histograms for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=15)  # Adjust bins as needed
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Calculate outlier limits for numerical columns
outlier_limits = calculate_outlier_limits(df, numerical_columns)

# Print the outlier limits for each numerical column and count the outliers
outlier_counts = {}
for column, (lower_limit, upper_limit) in outlier_limits.items():
    #print(f"Outlier limits for '{column}': Lower limit={lower_limit}, Upper limit={upper_limit}")
    lower_outliers = (df[column] < lower_limit).sum()
    upper_outliers = (df[column] > upper_limit).sum()
    total_outliers = lower_outliers + upper_outliers
    outlier_counts[column] = total_outliers
    #print(f"Number of outliers in '{column}': {total_outliers}")

# Treat outliers by replacing them with NaN values
for column, (lower_limit, upper_limit) in outlier_limits.items():
    if outlier_counts[column] > 0:
        df.loc[df[column] < lower_limit, column] = np.nan
        df.loc[df[column] > upper_limit, column] = np.nan

# Verify the outlier treatment with NaN values
outlier_counts_after_treatment = {}
for column, (lower_limit, upper_limit) in outlier_limits.items():
    lower_outliers_after = np.isnan(df[column]).sum()
    total_outliers_after = lower_outliers_after
    outlier_counts_after_treatment[column] = total_outliers_after
    #print(f"Number of outliers in '{column}' after treatment: {total_outliers_after}")
    

# Export DataFrame to a new CSV file
df.to_csv('loan_outliers_treated.csv', index=False)


#Replacing missing values or nan Values------------------------------------------------------------------------

def encode_dataframe(df):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.
    Parameters:
    df (DataFrame): The input DataFrame containing categorical columns to be encoded.
    Returns:
    encoded_df (DataFrame): A new DataFrame with categorical columns encoded.
    """
    encoded_df = df.copy()  # Make a copy of the original DataFrame to avoid modifying it
    
    # Iterate through each column in the DataFrame
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    
    return encoded_df

def preserve_no_nan_columns(df):
    """
    Create a duplicate DataFrame preserving columns with no NaN values and only the first column with NaN.
    Parameters:
    df (DataFrame): The input DataFrame.
    Returns:
    new_df (DataFrame): A new DataFrame with columns containing no NaN values and only the first column with NaN.
    """
    nan_detected = False
    columns_to_keep = []
    
    print("columns to check ", df.columns)
    
    for column in df.columns:
        print("now checking column ", column)

        if not df[column].isna().any():
            columns_to_keep.append(column)
                
        elif df[column].isna().any() and nan_detected:
            continue
        
        elif df[column].isna().any() and not nan_detected:
            nan_detected = True
            columns_to_keep.append(column)
        
    new_df = df[columns_to_keep].copy()
    return new_df


def get_nan_column(df):
    for column in df.columns:
        if df[column].isna().any():
            return column
    return False
            
# selected_dataset = pd.read_csv('rice-yield-act3.csv')
selected_dataset = pd.read_csv('loan_outliers_treated.csv')

# selected_dataset = pd.read_csv('dataset2.csv')

# stores the changes of each process
df = encode_dataframe(selected_dataset)        

# duplicate dataframe that is used for processing
df_copy = df


while(get_nan_column(df)): # use original df
    print('====================================================================')
    #print('nan column: ', get_nan_column(df)) # use original df
   # print('====================================================================')
    
    df_for_model = preserve_no_nan_columns(df) # dataframe with only one nan column, use original df
    column_to_predict = get_nan_column(df_for_model) 
    
    # Separate to train and test
    train_df = df_for_model.dropna(axis=0)
    test_df = df_for_model[df_for_model[column_to_predict].isnull()]
    
    # Create x and y train
    x_train = train_df.drop(column_to_predict, axis=1)
    y_train = train_df[column_to_predict]
    
    # Create x test
    x_test = test_df.drop(column_to_predict, axis=1)
    
    # Create the model
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    
    # Apply model
    y_pred = lr.predict(x_test)
    
    test_df[column_to_predict] = y_pred
    
    df_imputed = train_df.add(test_df, axis=1, fill_value=0)
    df[column_to_predict] = df_imputed[column_to_predict]
    
df.to_csv('cleaned_loan.csv', index=False)


