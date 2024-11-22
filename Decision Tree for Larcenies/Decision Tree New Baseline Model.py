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
crime_df.columns = crime_df.columns.str.strip()  # Remove leading/trailing spaces
crime_df.columns = crime_df.columns.str.replace('Ê', '', regex=False)  # Fix special characters

# Verify column names again
print("\nCleaned Columns in the dataset:")
print(crime_df.columns.tolist())

# Select columns of interest
features_to_keep = ['communityname', 'state', 'population', 'agePct12t21', 'agePct12t29', 
                    'agePct16t24', 'PctNotHSGrad', 'burglaries', 'burglPerPop', 'larcenies',
                    'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop']

# Ensure that only columns that actually exist in the DataFrame are selected
features_to_keep_existing = [col for col in features_to_keep if col in crime_df.columns]

# Filter the DataFrame to keep only the selected features
crime_df = crime_df[features_to_keep_existing]

# Display the first few rows of the dataset
print(crime_df.head())

# Replace "?" with NaN
crime_df.replace('?', pd.NA, inplace=True)

# Convert all columns to numeric except 'communityname' and 'state'
for column in crime_df.columns:
    if column not in ['communityname', 'state']:
        crime_df[column] = pd.to_numeric(crime_df[column], errors='coerce')

# Impute NaN values with the median of each column
for column in crime_df.columns:
    if column not in ['communityname', 'state']:
        median_value = crime_df[column].median()
        crime_df[column].fillna(median_value, inplace=True)

# Create a binary target variable based on the median of 'larcenies'
crime_df['larcenies_binary'] = np.where(crime_df['larcenies'] > crime_df['larcenies'].median(), 1, 0)

# Split the data into training and testing sets
features = ['population', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'PctNotHSGrad', 
            'burglaries', 'burglPerPop', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 
            'arsons', 'arsonsPerPop', 'nonViolPerPop']
X = crime_df[features]
y = crime_df['larcenies_binary']
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

# Visualizations

plt.figure(figsize=(8, 6))
crime_df['age_group'] = pd.cut(crime_df['agePct16t24'], bins=[0, 10, 20, 30, 40, 50], labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%"])
crime_grouped = crime_df.groupby('age_group', observed=True)['nonViolPerPop'].mean()
plt.plot(crime_grouped.index, crime_grouped.values, marker='o', linestyle='-', color='b')
plt.title('Non-Violent Crimes by Age Group (16-24 Shorter Range)')
plt.ylabel('Non-Violent Crimes Per Pop')
plt.xlabel('Age Group')
plt.show()

# Second Visual: Line Chart for Non-Violent Crimes by Age Group (Longer Range with Shading)
plt.figure(figsize=(8, 6))
crime_df['new_age_group'] = pd.cut(crime_df['agePct16t24'], bins=[0, 10, 20, 30, 40, 50, 60], labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%"])
crime_grouped_new = crime_df.groupby('new_age_group', observed=True)['nonViolPerPop'].mean()
plt.plot(crime_grouped_new.index, crime_grouped_new.values, marker='o', linestyle='--', color='r')
plt.title('Non-Violent Crimes by Age Group (16-24 Longer Range)')
plt.ylabel('Non-Violent Crimes Per Pop')
plt.xlabel('Age Group')
plt.axvspan('30-40%', '50-60%', color='red', alpha=0.3, label='After 45%')
plt.legend()
plt.show()

# Plot the decision tree
plt.figure(figsize=(40, 30))  # Significantly increase the plot size
plot_tree(c50_model, feature_names=features, class_names=['Low', 'High'], filled=True, rounded=True, fontsize=16)
plt.title("Decision Tree Visualization (C5.0 Algorithm)", fontsize=24)
plt.show()

# Display correlation matrix
correlation_matrix = crime_df[features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
plt.title('Heatmap of Feature Correlations', fontsize=16)
plt.show()

