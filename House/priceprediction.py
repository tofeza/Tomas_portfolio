import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("train.csv")

# Display the first few rows of the dataset
print(data.head())

# Select features and target variable
# Assuming 'SalePrice' is the target variable and the rest are features
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Check for non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
print(f'Non-numeric columns: {non_numeric_columns}')

# Handle missing values for non-numeric columns by filling with 'missing'
X[non_numeric_columns] = X[non_numeric_columns].fillna('missing')

# Convert categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X)

# Handle missing values for numeric columns by filling with mean
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Distribution of the target variable (SalePrice)
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted SalePrice')
plt.show()