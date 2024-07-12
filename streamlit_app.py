import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Function to compute CAGR and p-value
def compute_cagr(data, column):
    # Ensure data is a pandas DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Add a time variable, which is equally spaced
    data['Time'] = np.arange(1, len(data) + 1)
    
    # Apply natural log transformation to the target column
    data['LogColumn'] = np.log(data[column])
    
    # Perform the regression
    model = ols('LogColumn ~ Time', data=data).fit()
    
    # Compute CAGR
    cagr = np.exp(model.params['Time']) - 1
    
    # Extract p-value and adjusted R-squared
    p_value = model.pvalues['Time']
    adj_r_squared = model.rsquared_adj
    
    return cagr, p_value, adj_r_squared

# Function to compute mean, standard deviation, and coefficient of variation
def compute_statistics(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    cv_val = (std_val / mean_val)*100
    return mean_val, std_val, cv_val

# Function to compute CDVI
def compute_cdvi(cv, adj_r_squared):
    cdvi = cv * sqrt(1 - adj_r_squared)
    return cdvi

# Streamlit app
st.title('Trend Analyser by [Manojkumar Patil](https://github.com/patilmanojkumar)')
# Displaying the dynamic SVG banner
st.markdown(
    """
    <p align="center">
      <a href="https://github.com/DenverCoder1/readme-typing-svg">
        <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=yellow&size=30&center=true&vCenter=true&width=600&height=100&lines=Trend+Analysis+Made+Simple!;trend_analyser-1.0;" alt="Typing SVG">
      </a>
    </p>
    """,
    unsafe_allow_html=True
)

# File upload
uploaded_file = st.file_uploader("Upload a CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(data.head())
    
    column = st.selectbox("Select the target column", data.columns)
    
    if st.button('Compute CAGR and Statistics'):
        cagr, p_value, adj_r_squared = compute_cagr(data, column)
        mean_val, std_val, cv_val = compute_statistics(data, column)
        cdvi = compute_cdvi(cv_val, adj_r_squared)
        
        st.write(f"CAGR: {cagr:.2%}")
        st.write(f"P-Value: {p_value:.10f}")
        st.write(f"Mean: {mean_val:.2f}")
        st.write(f"Standard Deviation: {std_val:.2f}")
        st.write(f"Coefficient of Variation (CV): {cv_val:.2f}")
        st.write(f"Adjusted R Square: {adj_r_squared:.2f}")
        st.write(f"Cuddy Della Valle Index (CDVI): {cdvi:.2f}")

