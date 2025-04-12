import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\SHAID\Downloads\Lung Cancer Dataset.csv")

# Preview the data
print(df.head())
print("\nOriginal Columns in dataset:", df.columns.tolist())

# Rename columns for consistency
df.columns = [col.strip().upper().replace(' ', '_') for col in df.columns]
print("Updated Columns:", df.columns.tolist())

# Standardize string-type categorical data
if 'GENDER' in df.columns:
    df['GENDER'] = df['GENDER'].astype(str).str.strip().str.upper()

if 'LUNG_CANCER' in df.columns:
    df['LUNG_CANCER'] = df['LUNG_CANCER'].astype(str).str.strip().str.upper()

# 1. Distribution of Health Indicators
for col in ['AGE', 'ENERGY_LEVEL', 'OXYGEN_SATURATION']:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

# 2. Correlation Between Risk Factors
possible_risk_factors = ['SMOKING', 'POLLUTION', 'BREATHING_PROBLEM', 'BREATHING_ISSUE', 'POLLUTION_LEVEL']
available_risk_factors = [col for col in possible_risk_factors if col in df.columns]

if available_risk_factors:
    corr_matrix = df[available_risk_factors].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Matrix: Risk Factors')
    plt.show()
else:
    print("No valid risk factor columns found for correlation.")

# 3. Outlier Detection (Boxplots)
for col in ['AGE', 'ENERGY_LEVEL', 'OXYGEN_SATURATION']:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col], color='salmon')
        plt.title(f'Boxplot for {col}')
        plt.xlabel(col)
        plt.show()

# Boxplot of features by disease
if 'LUNG_CANCER' in df.columns:
    for col in ['AGE', 'ENERGY_LEVEL', 'OXYGEN_SATURATION']:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x='LUNG_CANCER', y=col, data=df, palette='Set2')
            plt.title(f'{col} vs Lung Cancer')
            plt.show()

# 4. Scatter & Pair Plots
if {'SMOKING', 'OXYGEN_SATURATION', 'LUNG_CANCER'}.issubset(df.columns):
    sns.scatterplot(x='SMOKING', y='OXYGEN_SATURATION', hue='LUNG_CANCER', data=df)
    plt.title('Smoking vs Oxygen Saturation by Disease Status')
    plt.show()

if {'ALCOHOL_CONSUMING', 'ENERGY_LEVEL', 'LUNG_CANCER'}.issubset(df.columns):
    sns.scatterplot(x='ALCOHOL_CONSUMING', y='ENERGY_LEVEL', hue='LUNG_CANCER', data=df)
    plt.title('Alcohol Consumption vs Energy Level by Disease Status')
    plt.show()

# Pairplot
pairplot_cols = ['SMOKING', 'ALCOHOL_CONSUMING', 'OXYGEN_SATURATION', 'ENERGY_LEVEL']
if all(col in df.columns for col in pairplot_cols + ['LUNG_CANCER']):
    sns.pairplot(df, vars=pairplot_cols, hue='LUNG_CANCER')
    plt.suptitle('Pairplot: Risk Factors vs Disease', y=1.02)
    plt.show()

# 5. Feature Distributions by Disease & Gender
if 'LUNG_CANCER' in df.columns:
    for feature in ['AGE', 'OXYGEN_SATURATION']:
        if feature in df.columns:
            plt.figure(figsize=(10, 5))
            sns.violinplot(x='LUNG_CANCER', y=feature, data=df, palette='Pastel1')
            plt.title(f'{feature} Distribution by Disease Status')
            plt.show()

# Gender-based differences in risk factors
if {'GENDER', 'SMOKING', 'LUNG_CANCER'}.issubset(df.columns):
    sns.catplot(x='GENDER', y='SMOKING', hue='LUNG_CANCER', kind='box', data=df, height=6, aspect=1.5)
    plt.title('Smoking by Gender and Disease Status')
    plt.show()

if {'GENDER', 'POLLUTION', 'LUNG_CANCER'}.issubset(df.columns):
    sns.catplot(x='GENDER', y='POLLUTION', hue='LUNG_CANCER', kind='violin', data=df, height=6, aspect=1.5)
    plt.title('Pollution Exposure by Gender and Disease Status')
    plt.show()
