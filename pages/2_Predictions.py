import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app import load_data, get_trained_model
from ml_model import predict_tip

st.title('Tip Predictions')

data = load_data()
model, encoders = get_trained_model()

# Create tabs for different prediction modes
tab1, tab2 = st.tabs(['Single Prediction', 'Batch Predictions'])

with tab1:
    st.subheader('Predict Individual Tip')
    col1, col2 = st.columns(2)
    
    with col1:
        total_bill = st.number_input('Total Bill Amount ($)', min_value=0.0, value=20.0)
        size = st.number_input('Party Size', min_value=1, value=2)
        sex = st.selectbox('Customer Gender', ['Male', 'Female'])
        month = st.selectbox('Month', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    with col2:
        smoker = st.selectbox('Smoker?', ['Yes', 'No'])
        day = st.selectbox('Day', ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
        time = st.selectbox('Time', ['Lunch', 'Dinner'])
        weather = st.selectbox('Weather', ['Sunny', 'Rainy', 'Cloudy', 'Snow'])
    
    if st.button('Predict Tip'):
        # Updated call to include month and weather
        predicted_tip = predict_tip(model, encoders, total_bill, size, sex, smoker, day, time, month, weather)
        st.success(f'Predicted Tip Amount: ${predicted_tip:.2f}')
        
        # Show tip percentage
        tip_percentage = (predicted_tip / total_bill) * 100
        st.info(f'Tip Percentage: {tip_percentage:.1f}%')
        
        # Show feature importance
        st.subheader('Feature Importance')
        # Updated feature list to include new features
        importance = pd.DataFrame({
            'Feature': ['Total Bill', 'Party Size', 'Gender', 'Smoker', 'Day', 'Time', 'Month', 'Weather', 'Is Weekend', 'Is Peak Season'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=importance, x='Importance', y='Feature')
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.subheader('Batch Predictions')
    st.write('Upload a CSV file with columns: total_bill, size, sex, smoker, day, time, month, weather')
    
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            # Updated required columns list
            required_columns = ['total_bill', 'size', 'sex', 'smoker', 'day', 'time', 'month', 'weather']
            
            if all(col in batch_data.columns for col in required_columns):
                # Make predictions
                predictions = []
                for _, row in batch_data.iterrows():
                    # Updated call to include month and weather
                    pred = predict_tip(model, encoders, 
                                     row['total_bill'], row['size'],
                                     row['sex'], row['smoker'],
                                     row['day'], row['time'],
                                     row['month'], row['weather'])
                    predictions.append(pred)
                
                batch_data['predicted_tip'] = predictions
                batch_data['tip_percentage'] = (batch_data['predicted_tip'] / 
                                              batch_data['total_bill']) * 100
                
                st.write('Predictions:')
                st.dataframe(batch_data)
                
                # Download predictions
                st.download_button(
                    label='Download Predictions as CSV',
                    data=batch_data.to_csv(index=False).encode('utf-8'),
                    file_name='tip_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error(f'CSV must contain columns: {', '.join(required_columns)}')
        except Exception as e:
            st.error(f'Error processing file: {str(e)}')