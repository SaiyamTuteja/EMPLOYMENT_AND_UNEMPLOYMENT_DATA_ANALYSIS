import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Read data from CSV files
df1 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df2 = pd.read_csv("Unemployment_in_India.csv")

# Merge dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# Perform label encoding on 'region' column
le = LabelEncoder()
merged_df['Region'] = le.fit_transform(merged_df['Region'])

# Define function for exploratory data analysis
def eda():
    # Exclude non-numeric column
    numeric_df = merged_df.select_dtypes(include=[np.number])
    
    # Display summary statistics
    print("Summary Statistics:")
    print(numeric_df.describe())
    
    # Display correlation matrix
    print("\nCorrelation Matrix:")
    print(numeric_df.corr())
    
    # Display heat map
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Heat Map of Correlation Matrix")
    plt.show()

#  function for random forest analysis

def random_forest(X, y):
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(X, y)
    y_pred = forest_reg.predict(X)
    r2 = r2_score(y, y_pred)
    print("R-squared: ", r2)

def visual_data():
    # Plot scatter plot of X and y
    X = merged_df[[' Estimated Labour Participation Rate (%)']]
    y = merged_df[' Estimated Unemployment Rate (%)']
    plt.figure(figsize=(10,6))
    plt.scatter(X, y, alpha=0.5)
    plt.xlabel(" Estimated Labour Participation Rate (%)")
    plt.ylabel(" Estimated Unemployment Rate (%)")
    plt.title("Unemployment Analysis: Labour Participation Rate vs. Unemployment Rate")
    plt.show()
    
    # Display histogram of unemployment rate
    plt.hist(merged_df[' Estimated Unemployment Rate (%)'], bins=20)
    plt.xlabel('Unemployment Rate (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Unemployment Rates in India')
    plt.show()
    # Display histogram of labour participation rate
    merged_df[' Estimated Labour Participation Rate (%)'].plot.hist(title="Labour Participation Rate Histogram")
    
    # Display heat map
    # sns.heatmap(merged_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title("Heat Map of Correlation Matrix")
    # plt.show()

X = merged_df[[' Estimated Labour Participation Rate (%)']]
y = merged_df[' Estimated Unemployment Rate (%)']

# Prompt user for input
while True:
    print("\nPlease select an option:")
    print("1. Explanatory Data Analysis")
    print("2. Random forest Analysis ")
    print("3. Visualized analysis")
    print("4. Exit Program")
    
    # Get user input
    choice = input("Enter choice number: ")
    
    # Perform selected analysis
    if choice == '1':
        eda()
    elif choice == '2':
        random_forest(X, y)
    elif choice == '3' :
        visual_data()
    elif choice == '4':
        break
    else:
        print("Invalid choice. Please enter a valid choice number.")
