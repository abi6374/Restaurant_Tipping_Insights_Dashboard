import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from app import load_data

st.title('Detailed Analysis')

data = load_data()

# Add filters in a cleaner layout
col1, col2 = st.columns(2)
with col1:
    day_filter = st.multiselect('Select Days', data['day'].unique())
    sex_filter = st.multiselect('Select Gender', data['sex'].unique())

with col2:
    time_filter = st.multiselect('Select Time', data['time'].unique())
    smoker_filter = st.multiselect('Smoker/Non-smoker', data['smoker'].unique())

# Filter data
filtered_data = data.copy()
if day_filter:
    filtered_data = filtered_data[filtered_data['day'].isin(day_filter)]
if time_filter:
    filtered_data = filtered_data[filtered_data['time'].isin(time_filter)]
if sex_filter:
    filtered_data = filtered_data[filtered_data['sex'].isin(sex_filter)]
if smoker_filter:
    filtered_data = filtered_data[filtered_data['smoker'].isin(smoker_filter)]

# Display key metrics
st.subheader('Key Metrics')
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
with metrics_col1:
    st.metric('Average Tip', f'${filtered_data["tip"].mean():.2f}')
with metrics_col2:
    st.metric('Average Bill', f'${filtered_data["total_bill"].mean():.2f}')
with metrics_col3:
    st.metric('Average Party Size', f'{filtered_data["size"].mean():.1f}')

# Advanced visualizations
st.subheader('Correlation Heatmap')
fig3, ax3 = plt.subplots(figsize=(10, 6))
numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(), annot=True, ax=ax3, cmap='coolwarm')
st.pyplot(fig3)
plt.close(fig3)

# Additional insights
st.subheader('Tips Distribution by Day and Time')
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.violinplot(x='day', y='tip', hue='time', data=filtered_data, ax=ax4)
st.pyplot(fig4)
plt.close(fig4)

# Show filtered data with download button
st.subheader('Filtered Data')
st.download_button(
    label='Download Filtered Data as CSV',
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='filtered_restaurant_data.csv',
    mime='text/csv'
)
st.dataframe(filtered_data)