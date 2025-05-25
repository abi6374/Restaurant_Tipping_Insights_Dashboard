import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ml_model import train_model, predict_tip
import io
import base64

# Load and enhance the tips dataset
@st.cache_data
def load_data():
    # Load original dataset
    data = sns.load_dataset('tips')
    
    # Create synthetic data for all days and months
    all_days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
    time_slots = ['Lunch', 'Dinner']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Calculate averages and standard deviations for existing data
    avg_bill = data['total_bill'].mean()
    std_bill = data['total_bill'].std()
    avg_tip = data['tip'].mean()
    std_tip = data['tip'].std()
    avg_size = data['size'].mean()
    
    # Additional features for synthetic data
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Snow']
    special_occasions = ['None', 'Birthday', 'Anniversary', 'Business', 'Holiday']
    reservation_status = ['Walk-in', 'Reserved']
    
    # Create synthetic data with more features
    synthetic_data = []
    for month in months:
        for day in all_days:
            for time in time_slots:
                # Generate more samples for weekends and seasonal variations
                base_samples = 40 if day in ['Sat', 'Sun'] else 20
                # Increase samples for peak seasons (summer and winter holidays)
                if month in ['Jun', 'Jul', 'Aug', 'Dec']:
                    num_samples = int(base_samples * 1.5)
                else:
                    num_samples = base_samples
                
                for _ in range(num_samples):
                    # Generate realistic bill amounts based on time, day, and season
                    base_bill = avg_bill * (1.2 if time == 'Dinner' else 1.0)
                    base_bill *= 1.3 if day in ['Fri', 'Sat'] else 1.0
                    # Seasonal adjustments
                    if month in ['Dec', 'Jul']:
                        base_bill *= 1.2  # Peak season markup
                    
                    # Add random variation
                    total_bill = np.clip(base_bill + np.random.normal(0, std_bill/2), 10, 100)
                    
                    # Generate realistic tip based on bill and other factors
                    base_tip_rate = 0.15 + np.random.normal(0, 0.03)
                    # Higher tips during holidays
                    if month == 'Dec':
                        base_tip_rate += 0.02
                    tip = total_bill * base_tip_rate
                    
                    # Weather based on month
                    if month in ['Dec', 'Jan', 'Feb']:
                        weather_odds = [0.2, 0.3, 0.2, 0.3]  # More snow and rain
                    elif month in ['Jun', 'Jul', 'Aug']:
                        weather_odds = [0.6, 0.2, 0.2, 0]    # More sunny days
                    else:
                        weather_odds = [0.4, 0.3, 0.3, 0]    # Balanced weather
                    
                    synthetic_data.append({
                        'total_bill': total_bill,
                        'tip': tip,
                        'sex': np.random.choice(['Male', 'Female']),
                        'smoker': np.random.choice(['Yes', 'No']),
                        'day': day,
                        'time': time,
                        'month': month,
                        'size': int(np.clip(np.random.normal(avg_size, 1), 1, 6)),
                        'weather': np.random.choice(weather_conditions, p=weather_odds),
                        'occasion': np.random.choice(special_occasions),
                        'reservation': np.random.choice(reservation_status)
                    })
    
    # Combine original and synthetic data
    synthetic_df = pd.DataFrame(synthetic_data)
    enhanced_data = pd.concat([data, synthetic_df], ignore_index=True)
    
    # Add derived features
    enhanced_data['tip_percentage'] = (enhanced_data['tip'] / enhanced_data['total_bill']) * 100
    enhanced_data['per_person_bill'] = enhanced_data['total_bill'] / enhanced_data['size']
    enhanced_data['is_weekend'] = enhanced_data['day'].isin(['Sat', 'Sun'])
    enhanced_data['is_peak_season'] = enhanced_data['month'].isin(['Jun', 'Jul', 'Aug', 'Dec'])
    
    return enhanced_data

data = load_data()

# Train ML model
@st.cache_data
def get_trained_model():
    return train_model(data)

model, encoders = get_trained_model()

# Title of the dashboard
st.title('Restaurant Tipping Insights Dashboard')
st.write('This dashboard analyzes tipping patterns in restaurants based on various factors.')

# Summary statistics with enhanced metrics
st.subheader('Key Metrics')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Average Tip', f'${data["tip"].mean():.2f}')

with col2:
    st.metric('Average Tip %', f'{data["tip_percentage"].mean():.1f}%')

with col3:
    st.metric('Average Bill', f'${data["total_bill"].mean():.2f}')

with col4:
    st.metric('Average Party Size', f'{data["size"].mean():.1f}')

# Enhanced visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader('Tips by Day and Time')
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='day', y='tip', hue='time', data=data, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader('Tip Percentage vs Bill Amount')
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x='total_bill', y='tip_percentage', hue='size', 
                    size='size', sizes=(50, 200), data=data, ax=ax2)
    ax2.set_xlabel('Total Bill ($)')
    ax2.set_ylabel('Tip Percentage (%)')
    st.pyplot(fig2)
    plt.close(fig2)

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'Detailed Analysis', 'Predictions', 'Trend Analysis', 'Export Data'])

if page == 'Overview':
    # Title of the dashboard
    st.title('Restaurant Tipping Insights Dashboard')
    st.write('This dashboard analyzes tipping patterns in restaurants based on various factors.')
    
    # Summary statistics
    st.subheader('Summary Statistics')
    st.write(data.describe())
    
    # Basic visualizations in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Tips by Sex')
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='sex', y='tip', data=data, ax=ax1)
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader('Total Bill vs Tip')
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x='total_bill', y='tip', hue='time', data=data, ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)

elif page == 'Detailed Analysis':
    st.title('Detailed Analysis')
    
    # Add filters
    st.sidebar.subheader('Filters')
    day_filter = st.sidebar.multiselect('Select Days', data['day'].unique())
    time_filter = st.sidebar.multiselect('Select Time', data['time'].unique())
    
    # Filter data
    filtered_data = data
    if day_filter:
        filtered_data = filtered_data[filtered_data['day'].isin(day_filter)]
    if time_filter:
        filtered_data = filtered_data[filtered_data['time'].isin(time_filter)]
    
    # Advanced visualizations
    st.subheader('Correlation Heatmap')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_data.corr(), annot=True, ax=ax3, cmap='coolwarm')
    st.pyplot(fig3)
    plt.close(fig3)
    
    # Additional insights
    st.subheader('Tips by Day and Time')
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='day', y='tip', hue='time', data=filtered_data, ax=ax4)
    st.pyplot(fig4)
    plt.close(fig4)
    
    # Show raw data
    st.subheader('Raw Data')
    st.write(filtered_data)

elif page == 'Predictions':
    st.title('Tip Prediction (ML Model)')
    
    # Advanced prediction form
    st.subheader('Predict Tip Amount')
    col1, col2 = st.columns(2)
    
    with col1:
        total_bill = st.number_input('Total Bill Amount', min_value=0.0, value=20.0)
        size = st.number_input('Party Size', min_value=1, value=2)
        sex = st.selectbox('Customer Sex', ['Male', 'Female'])
    
    with col2:
        smoker = st.selectbox('Smoker?', ['Yes', 'No'])
        day = st.selectbox('Day', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        time = st.selectbox('Time', ['Breakfast', 'Lunch', 'Dinner'])
    
    if st.button('Predict Tip'):
        predicted_tip = predict_tip(model, encoders, total_bill, size, sex, smoker, day, time)
        st.success(f'Predicted Tip Amount: ${predicted_tip:.2f}')
        
        # Show feature importance
        st.subheader('Feature Importance')
        importance = pd.DataFrame({
            'Feature': ['Total Bill', 'Party Size', 'Sex', 'Smoker', 'Day', 'Time'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

# Update the Trend Analysis page to show all days:
elif page == 'Trend Analysis':
    st.title('Trend Analysis')
    
    # Time-based analysis with all days
    st.subheader('Average Tips by Day and Time')
    
    # Create pivot table for average tips by day and time
    pivot_data = pd.pivot_table(
        data,
        values='tip',
        index='day',
        columns='time',
        aggfunc='mean'
    )
    
    # Reorder days to standard week order
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_data = pivot_data.reindex(day_order)
    
    # Plot heatmap of tips by day and time
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax5)
    plt.title('Average Tips by Day and Time')
    st.pyplot(fig5)
    plt.close(fig5)
    
    # Customer traffic analysis
    st.subheader('Customer Traffic by Day and Time')
    traffic_data = pd.crosstab([data['day']], data['time'])
    traffic_data = traffic_data.reindex(day_order)
    
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    traffic_data.plot(kind='bar', ax=ax6)
    plt.title('Customer Distribution')
    plt.xlabel('Day')
    plt.ylabel('Number of Parties')
    plt.legend(title='Time')
    plt.xticks(rotation=45)
    st.pyplot(fig6)
    plt.close(fig6)
    
    # Time-based analysis
    st.subheader('Average Tips by Day')
    daily_avg = data.groupby('day')['tip'].mean().reset_index()
    st.bar_chart(daily_avg.set_index('day'))
    
    # Tips percentage analysis
    st.subheader('Tip Percentage Distribution')
    data['tip_pct'] = data['tip'] / data['total_bill'] * 100
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='tip_pct', bins=30)
    plt.xlabel('Tip Percentage')
    st.pyplot(fig5)
    plt.close(fig5)
    
    # Busy hours analysis
    st.subheader('Customer Count by Day and Time')
    busy_hours = pd.crosstab(data['day'], data['time'])
    st.write(busy_hours)
    
    # Plot busy hours
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    busy_hours.plot(kind='bar', ax=ax6)
    plt.title('Customer Distribution')
    plt.xlabel('Day')
    plt.ylabel('Number of Parties')
    st.pyplot(fig6)
    plt.close(fig6)

elif page == 'Export Data':
    st.title('Export Data')
    
    # Add export options
    export_format = st.radio('Select Export Format', ['CSV', 'Excel'])
    
    if export_format == 'CSV':
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        st.markdown(f'<a href="{href}" download="tips_data.csv">Download CSV File</a>', unsafe_allow_html=True)
    else:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            data.to_excel(writer, sheet_name='Tips Data', index=False)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
        st.markdown(f'<a href="{href}" download="tips_data.xlsx">Download Excel File</a>', unsafe_allow_html=True)