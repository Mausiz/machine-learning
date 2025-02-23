# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:21:00 2025

@author: Metrics
"""

import pandas as pd

# Load dataset
df = pd.read_csv('data/predictive_maintenance.csv')

# Display first few rows
print(df.head())

# Get basic information about the dataset
print(df.info())

# Get statistical summary
print(df.describe())

# Check for missing values
print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for selected features
sns.pairplot(df[['feature1', 'feature2', 'target']])
plt.show()

"""2. Data preprocessing"""
# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
label_encoder = LabelEncoder()
df['categorical_column'] = label_encoder.fit_transform(df['categorical_column'])

from sklearn.preprocessing import StandardScaler

# Normalize the features
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""3. Saving"""
# Save the preprocessed data to new CSV file
df.to_csv('data/preprocessed_predictive_maintenance.csv', index=False)