import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Optional: Setup API key for OpenAI if using GPT models
api_key = st.secrets["openai_api_key"]

# Model choices directly list all models including Ollama's and GPT-4
model_choices = [
    'GPT-4', 'dolphin-mixtral', 'codegemma', 'llama2', 'mixtral', 'llava', 'mistral', 'llama3'
]

# Let users choose a model via radio buttons
model_choice = st.sidebar.radio("Choose an LLM for analysis:", model_choices)

# Ensure all logic using model_choice is placed after this point
st.title('Preliminary Data Analysis with LLMs')
data_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'pdf', 'json'])


# Function to load dataset based on the file type
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
                st.error("Unsupported file format")
        except Exception as e:
            st.error(f"An error occurred while loading the dataset: {e}")
    return None

# Visualization function
def display_visualizations(data):
    if isinstance(data, pd.DataFrame):
        st.header("Automated Data Visualizations")
        
        # Check for numeric data for Plotly visualizations
        if not data.empty and data.select_dtypes(include=[np.number]).columns.tolist():
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                selected_col = st.selectbox('Select a Numeric Column to Display with Plotly', numerical_cols)
                fig = px.histogram(data, x=selected_col)
                st.plotly_chart(fig, use_container_width=True)

                # Correlation heatmap if multiple numeric columns exist
                if len(numerical_cols) > 1:
                    corr_fig = px.imshow(data[numerical_cols].corr(), text_auto=True, aspect="auto")
                    st.plotly_chart(corr_fig, use_container_width=True)

        # Matplotlib/Seaborn visualizations for categorical data
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                if len(data[col].unique()) <= 10:  # Visualize only if not too many unique values
                    fig, ax = plt.subplots()
                    sns.countplot(x=col, data=data, palette='Set2')
                    st.pyplot(fig)

#Defining display_analysis_results
def display_analysis_results(prompt, response):
    output = response if isinstance(response, str) else 'No output generated'
    html_content = f"""
<style>
.analysis-card {{
    background-color: #f9f9f9;
    border-left: 5px solid #4CAF50;
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
    return html_content


# Single file uploader for multiple file types
if data_file is not None:
    dataset = load_dataset(data_file)
    if dataset is not None:
        columns = ", ".join(dataset.columns) if isinstance(dataset, pd.DataFrame) else "Text"
        st.write("Dataset loaded successfully!")
        question = st.text_input("Enter your question about the dataset:")

        if st.button('Analyze Data'):
            if model_choice == 'GPT-4':
                # GPT-4 specific logic
                agent = create_pandas_dataframe_agent(ChatOpenAI(api_key=api_key), df=dataset, model_name="gpt-4-turbo")
                response = agent.invoke(question)
                response_text = response.get('output', 'No response generated.')
            else:
                # Ollama models logic
                llm = Ollama(base_url='http://localhost:11434', model=model_choice)
                prompt_template = f"""
                You are a helpful assistant that can analyze and answer questions about a given dataset.
                The dataset is a table with the following columns: {columns}
                {{question}}
                """
                llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt_template, input_variables=["columns", "question"]))
                response_dict = llm_chain.invoke({"columns": columns, "question": question})
                response_text = response_dict.get("text", "No response generated.")

            result_html = display_analysis_results(question, response_text)
            st.markdown(result_html, unsafe_allow_html=True)
            display_visualizations(dataset)
    else:
        st.write("No dataset loaded or dataset could not be processed.")
else:
    st.write("Please upload a dataset to proceed.")




