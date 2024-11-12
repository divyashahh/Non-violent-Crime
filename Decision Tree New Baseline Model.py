import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Read dataset
crime_df = pd.read_csv(r"C:\Users\Ruby\Desktop\Predictive Analytics\Final Project\crimedata2 (1).csv", encoding='ISO-8859-1')

# Inspect the dataset
print("Columns in the dataset:")
print(crime_df.columns.tolist())  # Check for discrepancies

# Clean column names
crime_df.columns = crime_df.columns.str.strip()  # Removes leading/trailing spaces
crime_df.columns = crime_df.columns.str.replace('Ê', '', regex=False)  # Fix special characters

# Select columns of interest
features_to_keep = ['Êcommunityname', 'state', 'population', 'agePct12t21', 'agePct12t29', 
                    'agePct16t24', 'PctNotHSGrad', 'burglaries', 'burglPerPop', 'larcenies',
                    'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop']

# Ensure that only columns that actually exist in the DataFrame are selected
features_to_keep_existing = [col for col in features_to_keep if col in crime_df.columns]

# Filter the dataframe to keep only the selected features
crime_df = crime_df[features_to_keep_existing]

# Display the first few rows of the dataset
print(crime_df.head())

# Display the shape of the dataset
print(f"The dataset has {crime_df.shape[0]} rows and {crime_df.shape[1]} columns.")

# Replace "?" with NaN
crime_df.replace('?', pd.NA, inplace=True)

# Convert all columns to numeric except 'Êcommunityname' and 'state'
for column in crime_df.columns:
    if column not in ['Êcommunityname', 'state']:
        crime_df[column] = pd.to_numeric(crime_df[column], errors='coerce')

# Impute NaN values with the median of each column
for column in crime_df.columns:
    if column not in ['Êcommunityname', 'state']:
        median_value = crime_df[column].median()
        crime_df[column].fillna(median_value, inplace=True)

# Display the first few rows to verify imputation
print("\nDataset after imputation:")
print(crime_df.head())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(crime_df.describe())

# Display correlation matrix (excluding non-numeric columns)
numeric_columns = crime_df.select_dtypes(include=[np.number]).columns
correlation_matrix = crime_df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix)

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
plt.title('Heatmap of Feature Correlations', fontsize=16)
plt.show()

# Inspect and clean the column names
crime_df.comlumns = crime_df.columns.str.replace('Ê', '', regex=False)

# Split the data into training and testing sets
features = ['population', 'agePct12t21', 'PctNotHSGrad', 'burglaries', 'larcenies', 'autoTheft']
target = 'nonViolPerPop'

# Convert features to numeric and handles missing values
crime_df[features] = crime_df[features].apply(pd.to_numeric, errors='coerce')
crime_df.dropna(subset=features + [target], inplace=True)

#create a binary targetm_state=42)
crime_df['nonViolPerPop_binary'] = np.where(crime_df['nonViolPerPop'] > crime_df['nonViolPerPop'].median(), 1, 0)

#Spit the data into training and testing sets
X= crime_df[features]
y = crime_df['nonViolPerPop_binary']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and fit the Decision Tree model using the C5.0 algorithm (entropy criterion)
c50_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
c50_model.fit(x_train, y_train)

# Make predictions
y_pred_train = c50_model.predict(x_train)
y_pred_test = c50_model.predict(x_test)

# Evaluate Accuracy
print("Training Accuracy (C5.0):", accuracy_score(y_train, y_pred_train))
print("Testing Accuracy (C5.0):", accuracy_score(y_test, y_pred_test))

# Confusion Matrix
print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test, y_pred_test))

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test))

# Plot the decision tree
plt.figure(figsize=(40, 30))  # Significantly increase the plot size
plot_tree(c50_model, feature_names=features, class_names=['Low', 'High'], filled=True, rounded=True, fontsize=16)  # Increase font size for better readability
plt.title("Decision Tree Visualization (C5.0 Algorithm)", fontsize=24)  # Larger title font
plt.show()

# Display correlation matrix (excluding non-numeric columns)
numeric_columns = crime_df.select_dtypes(include=[np.number]).columns
correlation_matrix = crime_df[numeric_columns].corr()

