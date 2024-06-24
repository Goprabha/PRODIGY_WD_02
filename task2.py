import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'titanic.csv'  # Replace this with your actual file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data Cleaning

# Fill missing values in 'Age' with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing values in 'Fare' with the median fare
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop the 'Cabin' column as it has too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Convert 'Sex' and 'Embarked' to categorical variables
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# Display the cleaned dataset
print("\nCleaned dataset:")
print(df.head())

# Exploratory Data Analysis (EDA)

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Summary statistics for categorical columns
print("\nSummary statistics for categorical columns:")
print(df.describe(include=['category']))

# Visualizations

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Countplot of Survived
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=df)
plt.title('Count of Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Countplot of Sex
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex', data=df)
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Countplot of Embarked
plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked', data=df)
plt.title('Count of Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()

# Boxplot of Age by Survived
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age by Survived')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

# Boxplot of Fare by Survived
plt.figure(figsize=(10, 6))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare by Survived')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()

# Correlation matrix for numerical features only
numerical_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()