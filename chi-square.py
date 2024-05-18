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

# Drop the specified columns
columns_to_drop = [' income_annum', ' loan_amount', ' loan_term', ' cibil_score', 
                   ' residential_assets_value', ' commercial_assets_value', 
                   ' luxury_assets_value', ' bank_asset_value']
X = df.drop(columns=columns_to_drop).iloc[:,:-1]
Y = df.iloc[:,-1]

# Encode categorical features and target variable
def encode_dataframe(X):
    encoded_df = X.copy()
    for column in encoded_df.columns:
        if encoded_df[column].dtype == 'object':  # Check if the column is categorical
            label_encoder = LabelEncoder()  # Initialize LabelEncoder
            encoded_df[column] = label_encoder.fit_transform(encoded_df[column])  # Encode the column
    return encoded_df

encoded_X = encode_dataframe(X)
encoded_Y = LabelEncoder().fit_transform(Y)

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

# Export the filtered DataFrame to CSV
filtered_df.to_csv('chi_filtered_loan.csv', index=False)

# Display chi2_df in the variable explorer
chi2_df
