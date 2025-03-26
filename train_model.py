import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("modified_construction_delay_data.csv")

# Define categorical columns
categorical_columns = [
    "Project_Size", "Weather_Conditions", "Labor_Availability",
    "Material_Availability", "Site_Conditions", "Government_Approvals",
    "Budget_Issues", "Equipment_Availability"
]

# Apply Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders

# Split dataset into features and target variable
X = df.drop(columns=["Actual_Completion_Days"])
y = df["Actual_Completion_Days"]

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} days")

# Save trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save label encoders (only class mappings)
with open("label_encoders.pkl", "wb") as file:
    pickle.dump({col: le.classes_ for col, le in label_encoders.items()}, file)

print("Model and encoders saved successfully!")
