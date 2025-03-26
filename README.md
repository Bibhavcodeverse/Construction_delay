# Construction Delay Prediction

## Overview
This project predicts **construction delays** using a **Random Forest Regressor**. Users can input project details via a **Streamlit web app**, and the trained model will estimate the expected completion time.

## Features
- **Trains a Random Forest model** to predict construction delays.
- **Encodes categorical features** using Label Encoding.
- **Deploys a web interface** with Streamlit for easy input and predictions.
- **Saves & loads trained models** for future predictions.

## Dependencies
Ensure you have the following installed:

```bash
pip install pandas numpy scikit-learn streamlit pickle-mixin
```

## How to Run
### 1. Train the Model
```bash
python train_model.py
```
- Loads `modified_construction_delay_data.csv`.
- Encodes categorical variables.
- Trains a **Random Forest Regressor**.
- Saves the model as `model.pkl` and encoders as `label_encoders.pkl`.

### 2. Run the Web App
```bash
streamlit run app.py
```
- Loads the trained model and label encoders.
- Provides a user-friendly interface to enter project details.
- Predicts the expected **completion time** in days.

## Dataset Requirements
The dataset should include:
- **Categorical features** (e.g., Project Size, Weather Conditions, Labor Availability, etc.).
- **Target variable**: `Actual_Completion_Days` (numeric value).

## Expected Output
- **Model Evaluation**: The training script prints **Mean Absolute Error (MAE)**.
- **Prediction UI**: The app outputs the estimated project completion time based on user inputs.

## Notes
- Ensure `modified_construction_delay_data.csv` is in the correct location before training.
- Modify hyperparameters in `train_model.py` to improve performance.



