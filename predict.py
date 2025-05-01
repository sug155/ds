# 1. Import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load dataset
df = pd.read_csv('AAPL.csv')  # Make sure this file exists in the same directory

# 3. Descriptive Analysis
print("=== Descriptive Statistics ===")
print(df.describe())

print("\n=== Dataset Info ===")
print(df.info())

# 4. Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 5. Drop non-numerical or irrelevant columns like 'Date' if present
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])

# Ensure all columns are numeric (convert if needed)
df = df.select_dtypes(include=[np.number])

# 6. Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(df,hue='Close')
plt.show()

X = df.drop('Close', axis=1)
y = df['Close']

# 8. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 10. Regression Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Support Vector Regressor': SVR()
}

results = []

# 11. Train, Predict, and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': round(mae, 2),
        'MSE': round(mse, 2),
        'RMSE': round(rmse, 2),
        'R2 Score': round(r2, 2)
    })

    print(f"\n=== {name} ===")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# 12. Summary Table
results_df = pd.DataFrame(results)
print("\n=== Regression Model Comparison ===")
print(results_df)

# 13. Visualization
results_df.set_index('Model')[['MAE', 'RMSE', 'R2 Score']].plot(kind='bar', figsize=(12, 6))
plt.title('Regression Model Performance on AAPL Stock')
plt.ylabel('Metric Value')
plt.grid(True)
plt.tight_layout()
plt.show()
