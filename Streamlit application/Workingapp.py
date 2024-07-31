import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import docx
import xml.etree.ElementTree as ET
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Access API key securely
api_key = st.secrets["openai_api_key"]



# Define the display_analysis_results function early to ensure it's available when needed
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

st.title('Data Analysis with LLMs')
st.header('Upload your data')
data_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'txt', 'html', 'xml', 'pdf', 'png', 'jpeg', 'docx'], key='unique_file_uploader')

def load_dataset(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.type
        try:
            if file_type == "text/csv" or file_type == "application/vnd.ms-excel":
                return pd.read_csv(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(uploaded_file)
            elif file_type == "text/plain" or file_type == "text/html":
                return uploaded_file.getvalue().decode("utf-8")
            elif file_type == "text/xml":
                tree = ET.parse(uploaded_file)
                return ET.tostring(tree.getroot(), encoding='utf8', method='xml')
            elif file_type == "application/pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    pages = [page.extract_text() for page in pdf.pages]
                return "\n".join(pages)
            elif file_type in ["image/jpeg", "image/png"]:
                return Image.open(uploaded_file)
            elif file_type is "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            else:
                return "Unsupported file format"
        except Exception as e:
            st.error(f"An error occurred while loading the dataset: {e}")
            return None

dataset = load_dataset(data_file)
agent = None  # Define agent as None by default

if dataset is not None:
    if isinstance(dataset, pd.DataFrame) or isinstance(dataset, str):
        agent = create_pandas_dataframe_agent(ChatOpenAI(), df=dataset, model_name="gpt-4-turbo")
        custom_prompt = st.text_area("Enter a custom prompt for analysis:", value="Please summarize the content or provide insights based on the analysis.")
        if st.button('Analyze') and agent:
            try:
                response = agent.invoke(custom_prompt, handle_parsing_errors=True)
                if 'output' in response:
                    st.markdown(display_analysis_results(custom_prompt, response['output']), unsafe_allow_html=True)
                else:
                    st.error("Failed to parse the output or no output was generated.")
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

        # Automated Data Visualizations
        st.header("Automated Data Visualizations")
        if isinstance(dataset, pd.DataFrame):
            numerical_cols = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numerical_cols:
                for col in numerical_cols:
                    st.subheader(f'Distribution of {col}')
                    fig, ax = plt.subplots()
                    sns.histplot(dataset[col], kde=True, ax=ax)
                    st.pyplot(fig)

                if len(numerical_cols) > 1:
                    st.subheader('Correlation Heatmap')
                    fig, ax = plt.subplots()
                    sns.heatmap(dataset[numerical_cols].corr(), annot=True, cmap='viridis', ax=ax)
                    st.pyplot(fig)

            # Categorical Data Visualization
            categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                for col in categorical_cols:
                    if len(dataset[col].unique()) <= 10:  # Adjust threshold as needed
                        st.subheader(f'Count Plot for {col}')
                        fig, ax = plt.subplots()
                        sns.countplot(x=col, data=dataset, palette='Set3', ax=ax)
                        st.pyplot(fig)
elif isinstance(dataset, Image.Image):
    st.success('File successfully uploaded and read as an image.')
    st.image(dataset, caption='Uploaded Image')
    st.info("Image analysis not currently supported with LangChain.")
else:
    st.error('Unsupported file format or unable to load the file correctly.')
