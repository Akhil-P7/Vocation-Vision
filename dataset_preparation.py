import pandas as pd

# Load the dataset
df = pd.read_csv("jobs.csv")

# Inspect the dataset
print("Columns:", df.columns)
print("First 5 rows:")
print(df.head())

# Combine relevant fields (adjust these fields as per your dataset)
df['combined'] = df['Job Title'] + " " + df['Skills'] + " " + df['Location']

# Handle missing values
df['combined'] = df['combined'].fillna('')

# Display cleaned data
print("Cleaned Data:")
print(df[['combined']].head())
