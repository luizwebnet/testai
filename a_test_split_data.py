import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
df = pd.read_csv('medical_qa.csv')

# Display dataset information
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df['tag'].value_counts())

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    df['question'], df[['answer', 'tag']], test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
