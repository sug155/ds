# Import required libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset
df = pd.read_csv('Iris.csv')  # <-- use df instead of data

# 2. Descriptive Analysis
print("=== Descriptive Statistics ===")
print(df.describe())
print("\n=== Dataset Info ===")
print(df.info())
print("\n=== Class Distribution ===")
print(df['Species'].value_counts())  # <-- 'Species' not 'target'

# 3. Data Pre-processing
# Check missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Drop ID column if exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Encoding target labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Separate features and target
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Species', axis=1))  # <-- drop 'Species'
y = df['Species']  # <-- target is 'Species'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Data Visualization

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df, hue='Species')
plt.show()

# 5. Correlation and Covariance Analysis
print("\n=== Correlation Matrix ===")
print(df.corr())

print("\n=== Covariance Matrix ===")
print(df.cov())

# 6. Build Classification Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and Evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Final metrics summary
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)
