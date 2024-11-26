# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
crime_df = pd.read_csv(r"C:\Users\Ruby\Desktop\Predictive Analytics\Final Project\crimedata2 (1).csv", encoding='ISO-8859-1')

# Display the column names in the dataset
print("Columns in the dataset:")
print(crime_df.columns.tolist())

# Clean column names
crime_df.columns = crime_df.columns.str.strip().str.replace('Ê', '', regex=False)

# Define features to keep
features_to_keep = [
    'communityname', 'state', 'population', 'agePct12t21', 'agePct12t29', 'agePct16t24', 
    'PctNotHSGrad', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 
    'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop', 'racepctblack', 
    'racePctWhite', 'racePctAsian', 'racePctHisp', 'PctUnemployed', 'medIncome'
]

# Filter existing columns
crime_df = crime_df[[col for col in features_to_keep if col in crime_df.columns]]

# Replace '?' with NaN
crime_df.replace('?', pd.NA, inplace=True)

# Convert relevant columns to numeric
for column in crime_df.columns:
    if column not in ['communityname', 'state']:
        crime_df[column] = pd.to_numeric(crime_df[column], errors='coerce')

# Impute missing values with the median
for column in crime_df.columns:
    if column not in ['communityname', 'state']:
        crime_df[column].fillna(crime_df[column].median(), inplace=True)

# Create binary target for larcenies
crime_df['larcenies_binary'] = np.where(crime_df['larcenies'] > crime_df['larcenies'].median(), 1, 0)

# Categorize numerical features
crime_df['Burglaries_Category'] = pd.cut(crime_df['burglaries'], bins=[-float('inf'), 95, 507.5, float('inf')], labels=['Low', 'Medium', 'High'])
crime_df['Larcenies_Category'] = pd.cut(crime_df['larcenies'], bins=[-float('inf'), 392.5, 1673, float('inf')], labels=['Low', 'Medium', 'High'])
crime_df['AutoTheft_Category'] = pd.cut(crime_df['autoTheft'], bins=[-float('inf'), 30, 231.5, float('inf')], labels=['Low', 'Medium', 'High'])

# Diversity classification
def classify_diversity(racepctblack, racePctWhite, racePctAsian, racePctHisp):
    if max(racepctblack, racePctWhite, racePctAsian, racePctHisp) > 90:
        return 'Low Diversity'
    elif max(racepctblack, racePctWhite, racePctAsian, racePctHisp) > 70:
        return 'Medium Diversity'
    else:
        return 'High Diversity'

crime_df['diversity_level'] = crime_df.apply(lambda row: classify_diversity(
    row['racepctblack'], row['racePctWhite'], row['racePctAsian'], row['racePctHisp']), axis=1)

# Additional categorization
crime_df['UnemploymentCategory'] = pd.cut(crime_df['PctUnemployed'], bins=[0, 5, 15, float('inf')], labels=['Low', 'Medium', 'High'])
crime_df['medIncome_Category'] = pd.cut(crime_df['medIncome'], bins=[-float('inf'), 30000, 70000, float('inf')], labels=['Low', 'Medium', 'High'])
crime_df['Population_Category'] = pd.cut(crime_df['population'], bins=[-float('inf'), 14366, 43024, float('inf')], labels=['Low', 'Medium', 'High'])

# One-hot encoding for categorical features
crime_df = pd.get_dummies(crime_df, columns=[
    'Burglaries_Category', 'Larcenies_Category', 'AutoTheft_Category', 'diversity_level', 
    'UnemploymentCategory', 'Population_Category'
])

# Prepare training and testing sets
features = [
    'population', 'agePct12t21', 'PctNotHSGrad', 'nonViolPerPop', 
    'Burglaries_Category_Low', 'Burglaries_Category_Medium', 'Burglaries_Category_High',
    'diversity_level_High Diversity', 'diversity_level_Medium Diversity', 
    'UnemploymentCategory_Low', 'UnemploymentCategory_Medium', 'UnemploymentCategory_High',
    'Population_Category_Low', 'Population_Category_Medium', 'Population_Category_High'
]
X = crime_df[features]
y = crime_df['larcenies_binary']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a Decision Tree model
c50_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
c50_model.fit(x_train, y_train)

# Evaluate the model
y_pred_test = c50_model.predict(x_test)
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test))

# Plot the decision tree
plt.figure(figsize=(40, 30))
plot_tree(c50_model, feature_names=features, class_names=['Low', 'High'], filled=True, rounded=True, fontsize=16)
plt.title("Decision Tree Visualization (C5.0 Algorithm)", fontsize=24)
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(crime_df['nonViolPerPop'], crime_df['larcenies'], alpha=0.6, color='blue')
plt.title('Scatter Plot: Non-Violent Crimes Per Population vs. Larcenies', fontsize=16)
plt.xlabel('Non-Violent Crimes Per Population', fontsize=14)
plt.ylabel('Larcenies', fontsize=14)
plt.grid(True)
plt.show()

## 4. Scatter Plot: Age (16-24) vs Non-Violent Crimes
plt.figure(figsize=(10, 6))
plt.scatter(crime_df['agePct16t24'], crime_df['nonViolPerPop'], alpha=0.6, color='orange')
plt.title('Scatter Plot: Age Percentage (16-24) vs. Non-Violent Crimes Per Population', fontsize=16)
plt.xlabel('Age Percentage (16-24)', fontsize=14)
plt.ylabel('Non-Violent Crimes Per Population', fontsize=14)
plt.grid(True)
plt.show()

## 5. Box Plot: Non-Violent Crimes by Population Density
plt.figure(figsize=(12, 6))
crime_df['PopulationDensityCategory'] = crime_df['agePct16t24'].apply(
    lambda x: 'High Populated Dense Area' if x >= 14.345 else 'Not High Populated Dense Area'
)
sns.boxplot(x='PopulationDensityCategory', y='nonViolPerPop', data=crime_df)
plt.title('Box Plot of Non-Violent Crimes by Population Density Category', fontsize=16)
plt.xlabel('Population Density Category', fontsize=12)
plt.ylabel('Non-Violent Crimes Per Population', fontsize=12)
plt.show()

