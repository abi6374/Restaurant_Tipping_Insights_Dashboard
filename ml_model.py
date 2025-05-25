from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def prepare_features(data):
    # Create encoders for categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_day = LabelEncoder()
    le_time = LabelEncoder()
    le_month = LabelEncoder()
    le_weather = LabelEncoder()
    
    # Create feature matrix
    X = pd.DataFrame()
    X['total_bill'] = data['total_bill']
    X['size'] = data['size']
    X['sex'] = le_sex.fit_transform(data['sex'])
    X['smoker'] = le_smoker.fit_transform(data['smoker'])
    X['day'] = le_day.fit_transform(data['day'])
    X['time'] = le_time.fit_transform(data['time'])
    X['month'] = le_month.fit_transform(data['month'])
    X['weather'] = le_weather.fit_transform(data['weather'])
    X['is_weekend'] = data['is_weekend']
    X['is_peak_season'] = data['is_peak_season']
    
    return X, (le_sex, le_smoker, le_day, le_time, le_month, le_weather)

def predict_tip(model, encoders, total_bill, size, sex, smoker, day, time, month, weather):
    le_sex, le_smoker, le_day, le_time, le_month, le_weather = encoders
    
    # Prepare single prediction
    X_pred = pd.DataFrame({
        'total_bill': [total_bill],
        'size': [size],
        'sex': le_sex.transform([sex]),
        'smoker': le_smoker.transform([smoker]),
        'day': le_day.transform([day]),
        'time': le_time.transform([time]),
        'month': le_month.transform([month]),
        'weather': le_weather.transform([weather]),
        'is_weekend': [day in ['Sat', 'Sun']],
        'is_peak_season': [month in ['Jun', 'Jul', 'Aug', 'Dec']]
    })
    
    return model.predict(X_pred)[0]

def train_model(data):
    X, encoders = prepare_features(data)
    y = data['tip']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoders