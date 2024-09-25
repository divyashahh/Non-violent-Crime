# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file with a specified encoding
crime_df = pd.read_csv(r"C:\Users\krist\OneDrive\Desktop\D&S3\crimedata2.csv", encoding='ISO-8859-1')

# Display the column names in the dataset
print("Columns in the dataset:")
print(crime_df.columns)

# Define the features to keep (without special characters)
features_to_keep = [
    'communityname', 'state', 'population', 'agePct12t21', 'agePct12t29', 
    'agePct16t24', 'PctNotHSGrad', 'burglaries', 'burglarPerPop', 'larcenies', 
    'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 
    'nonViolPerPop'
]

# Ensure that only columns that actually exist in the DataFrame are selected
features_to_keep_existing = [col for col in features_to_keep if col in crime_df.columns]

# Filter the dataframe to keep only the selected features
crime_df = crime_df[features_to_keep_existing]

# Display the first few rows of the dataset
print(crime_df.head())

# Display the shape of the dataset
print(f"The dataset has {crime_df.shape[0]} rows and {crime_df.shape[1]} columns.")

# Display data types of the columns
print("\nData types of columns:")
print(crime_df.dtypes)

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

# Display the first few rows to verify imputation
print(crime_df.head())

# Display the shape of the dataset after imputation
print(f"The dataset after imputation has {crime_df.shape[0]} rows and {crime_df.shape[1]} columns.")

# Display descriptive statistics
print("\nDescriptive statistics:")
print(crime_df.describe())

# Display correlation matrix
print("\nCorrelation matrix:")
print(crime_df.corr())

# Comment on the percentage threshold for higher density locations
print("\nNote: In our research question, we consider a location with a higher density of individuals aged 16-24 to be one where the percentage of this age group is at least 14.345%. This threshold is based on the fact that 75% of the data has at least 14.345% of individuals aged 16-24 living in those locations.")

# Display the first few rows of the selected columns
print("\nFirst few rows of 'agePct16t24':")
print(crime_df[['agePct16t24']].head())

# Calculate the average of the 'agePct16t24' column
average_agePct16t24 = crime_df['agePct16t24'].mean()
print("\nThe average percentage of individuals aged 16-24 is:", average_agePct16t24)

# Find the maximum value in the 'agePct16t24' column
max_agePct16t24 = crime_df['agePct16t24'].max()
print("The maximum percentage of individuals aged 16-24 is:", max_agePct16t24)

# Set pandas to display more rows (for example, 20 rows)
pd.set_option('display.max_rows', 20)

# Display the first 20 rows of the 'nonViolPerPop' column
print("\nFirst 20 rows of 'nonViolPerPop':")
print(crime_df[['nonViolPerPop']].head(20))

# Add a new column 'PopulationDensityCategory'
crime_df['PopulationDensityCategory'] = crime_df['agePct16t24'].apply(
    lambda x: 'High Populated Dense Area' if x >= 14.345 else 'Not High Populated Dense Area'
)

# Display the first few rows to verify the new column
print("\nFirst few rows with the new column 'PopulationDensityCategory':")
print(crime_df.head())

# Plot 'nonViolPerPop' against 'agePct16t24'
plt.figure(figsize=(10, 6))
plt.scatter(crime_df['nonViolPerPop'], crime_df['agePct16t24'], color='blue', alpha=0.5)
plt.title('Relationship between Non-Violent Crimes and Age 16-24 Percent', fontsize=14)
plt.xlabel('Non-Violent Crimes per Population', fontsize=12)
plt.ylabel('Percentage of Population Aged 16-24', fontsize=12)
plt.show()
