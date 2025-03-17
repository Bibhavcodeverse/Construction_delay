# Construction_delay

#code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("construction_delay_data.csv")

# Display first few rows
print("Dataset Sample:\n", df.head())

# Encode categorical variables using Label Encoding
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
    label_encoders[col] = le  # Save encoders for future use

# Split dataset into features and target variable
X = df.drop(columns=["Actual_Completion_Days"])
y = df["Actual_Completion_Days"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} days")

# Save trained model and label encoders
joblib.dump(model, "construction_delay_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and encoders saved successfully!")
