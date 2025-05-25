import streamlit as st
import pandas as pd
from app import load_data
import io
import base64

st.title('Export Data')

data = load_data()

# Add data filtering options
st.subheader('Filter Data Before Export')

col1, col2 = st.columns(2)
with col1:
    days = st.multiselect('Select Days', data['day'].unique())
    times = st.multiselect('Select Times', data['time'].unique())

with col2:
    min_bill = st.number_input('Minimum Bill Amount', value=float(data['total_bill'].min()))
    max_bill = st.number_input('Maximum Bill Amount', value=float(data['total_bill'].max()))

# Filter data
filtered_data = data.copy()
if days:
    filtered_data = filtered_data[filtered_data['day'].isin(days)]
if times:
    filtered_data = filtered_data[filtered_data['time'].isin(times)]
filtered_data = filtered_data[
    (filtered_data['total_bill'] >= min_bill) &
    (filtered_data['total_bill'] <= max_bill)
]

# Show preview
st.subheader('Data Preview')
st.dataframe(filtered_data.head())

# Export options
st.subheader('Export Options')

col3, col4 = st.columns(2)

with col3:
    # CSV export
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label='Download as CSV',
        data=csv,
        file_name='restaurant_data.csv',
        mime='text/csv'
    )

with col4:
    # Excel export
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        filtered_data.to_excel(writer, sheet_name='Restaurant Data', index=False)
    
    st.download_button(
        label='Download as Excel',
        data=buffer.getvalue(),
        file_name='restaurant_data.xlsx',
        mime='application/vnd.ms-excel'
    )

# Show data statistics
st.subheader('Data Statistics')
col5, col6, col7 = st.columns(3)

with col5:
    st.metric('Total Records', len(filtered_data))

with col6:
    st.metric('Average Bill', f'${filtered_data["total_bill"].mean():.2f}')

with col7:
    st.metric('Average Tip', f'${filtered_data["tip"].mean():.2f}')