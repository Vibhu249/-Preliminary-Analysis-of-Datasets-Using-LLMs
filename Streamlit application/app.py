import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import docx
import xml.etree.ElementTree as ET
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # PyMuPDF
import logging
from io import StringIO
import plotly.express as px
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


# Access API key securely
api_key = st.secrets["openai_api_key"]

# Global variable for balance
initial_balance = 6.74

# Utility function to display analysis results
def display_analysis_results(prompt, response):
    output = response.get('output', 'No output generated') if isinstance(response, dict) else response
    return f"""
        <style>
            .analysis-card {{
                background-color: #f9f9f9;
                border-left: 5px solid #4CAF50;
                margin-bottom: 10px;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .analysis-header {{
                font-size: 20px;
                color: #333;
                margin-bottom: 15px;
                font-weight: bold;
            }}
            .analysis-content {{
                font-size: 16px;
                color: #555;
            }}
        </style>
        <div class="analysis-card">
            <div class="analysis-header">LangChain Analysis Input</div>
            <div class="analysis-content">{prompt}</div>
        </div>
        <div class="analysis-card">
            <div class="analysis-header">LangChain Analysis Output</div>
            <div class="analysis-content">{output}</div>
        </div>
    """

st.title('Preliminary Data Analysis with LLMs')
data_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'pdf', 'json'], key='unique_file_uploader')

def load_dataset(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.type
        try:
            if file_type == "application/json":
                return pd.read_json(uploaded_file)
            elif file_type == "text/csv" or file_type == "application/vnd.ms-excel":
                return pd.read_csv(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(uploaded_file)
            elif file_type == "application/pdf":
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                return pd.DataFrame([text], columns=['Text'])
            else:
                return "Unsupported file format"
        except Exception as e:
            st.error(f"An error occurred while loading the dataset: {e}")
            return None

dataset = load_dataset(data_file)
if dataset is not None:
    config = {"verbose": True}
    agent = create_pandas_dataframe_agent(ChatOpenAI(api_key=api_key), df=dataset, model_name="gpt-4-turbo", config=config)
    custom_prompt = st.text_area("Enter a custom prompt for analysis:")
    if st.button('Analyze') and agent:
        response = agent.invoke(custom_prompt)
        st.markdown(display_analysis_results(custom_prompt, response), unsafe_allow_html=True)

# Remaining code for token calculations and visualizations would go here

# Token cost calculations
def calculate_and_display_cost(total_tokens):
    cost_per_1000_tokens = 0.002  # Cost per 1000 tokens, this should be adjusted based on your specific pricing model
    cost = (total_tokens / 1000) * cost_per_1000_tokens
    initial_balance -= cost
    st.sidebar.write(f"Tokens used: {total_tokens}")
    st.sidebar.write(f"Cost of this query: ${cost:.4f}")
    st.sidebar.write(f"Remaining balance: ${initial_balance:.2f}")

# Visualization section with both Matplotlib/Seaborn and Plotly
def display_visualizations(data):
    if isinstance(data, pd.DataFrame):
        st.header("Automated Data Visualizations")

        # Using Plotly for interactive visualizations
        if not data.empty and data.select_dtypes(include=[np.number]).columns.tolist():
            st.subheader('Interactive Plotly Charts')

            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox('Select Column to Display with Plotly', numerical_cols)
            fig = px.histogram(data, x=selected_col)
            st.plotly_chart(fig, use_container_width=True)

            if len(numerical_cols) > 1:
                st.subheader('Interactive Correlation Heatmap with Plotly')
                corr_fig = px.imshow(data[numerical_cols].corr(), text_auto=True, aspect="auto")
                st.plotly_chart(corr_fig, use_container_width=True)

        # Continue with Seaborn and Matplotlib visualizations as previously implemented
        if 'object' in data.dtypes or 'category' in data.dtypes:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                for col in categorical_cols:
                    if len(data[col].unique()) <= 10:  # Visualize only if there are not too many unique values
                        fig, ax = plt.subplots()
                        sns.countplot(x=col, data=data, palette='Set2')
                        st.pyplot(fig)

# Handling various file types and displaying content
if isinstance(dataset, pd.DataFrame):
    st.dataframe(dataset.head())  # Display the first few rows of the DataFrame
    display_visualizations(dataset)

elif isinstance(dataset, str):  # For text data from PDFs or other text files
    st.text_area("Text extracted from file", dataset, height=300)
    # If more processing is needed for text analysis, it can be added here

elif isinstance(dataset, Image.Image):
    st.success('File successfully uploaded and read as an image.')
    st.image(dataset, caption='Uploaded Image', use_column_width=True)
    st.info("Image analysis not currently supported with LangChain.")
else:
    st.error('Unsupported file format or unable to load the file correctly.')

# This assumes your app handles multiple file types and the conditions handle each type accordingly

