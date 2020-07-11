import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


# Describe the data frame
print(df.describe())
print(df.describe(include=['object']))
print(df.info())


# Drooping the Car name as it will not have bg impact
final_data = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
print(final_data.head())

# Creating new feature
final_data['current_year'] = 2020
final_data['no_of_years'] = final_data['current_year'] - final_data['Year']
print(final_data.head())


# Dropping the columns year and current_year
final_data.drop(['current_year', 'Year'], inplace=True, axis=1)
print(final_data.head())


# convert categorical variables into numeric columns
# drop_first was set as True so that dummy variable trap could be avoided
final_data = pd.get_dummies(final_data, drop_first=True)
print(final_data.head())

# Finding the correlation
print(final_data.corr())

# Correlation plot
# print(sns.pairplot(final_data))
# print(plt.show())

# Plotting heat map
# Here correlation uses pearson correlation
plt.figure(figsize=(15, 15))
sns.heatmap(final_data.corr(), annot=True)
plt.tight_layout()
# print(plt.show())


# Splitting into dependent and independent features
X = final_data.iloc[:, 1:]
Y = final_data.iloc[:, 0]
print(X.head())
print(type(X))
print(Y.head())
print(type(Y))

# Feature Importance

from sklearn.ensemble import ExtraTreesRegressor

extra_tree_regressor = ExtraTreesRegressor()
extra_tree_regressor.fit(X, Y)

print(extra_tree_regressor.feature_importances_)
