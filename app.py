import pandas as pd

# Reading the file as df
df = pd.read_csv('car data.csv')

# Df read the 5 rows
print(df.head())

# DF find the no of columns and rows
print(df.shape)

# List of Columns
print(list(df.columns))

# Unique Values for Categorical columns
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df['Fuel_Type'].unique())



# Find the missing values
print(df.isnull().sum())


# Describe the dataframe
print(df.describe())
print(df.describe(
    include=['object']
))
print(df.info())


# Drooping the Car name as it will not have bg impact
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]

print(final_dataset.head())