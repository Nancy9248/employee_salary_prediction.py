# Employee Salary Prediction Using Random Forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("adult 3.csv")

# Replace '?' with NaN
data = data.replace("?", pd.NA)

# Drop rows with missing values
data = data.dropna()

# Strip whitespace
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include="object"):
    data[col] = data[col].str.strip()

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include="object"):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data.drop("income", axis=1)
y = data["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
