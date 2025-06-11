# Preliminary Analysis of Datasets Using LLMs

This repository contains all the necessary files for the preliminary analysis of datasets using large language models (LLMs). 
Below are the details of each folder and file.

![Flow Chart](https://github.com/Vibhu249/-Preliminary-Analysis-of-Datasets-Using-LLMs/blob/main/FlowChart.png)

### Introduction

You can use this project to:  
• Run LLM-driven summaries or suggestions on your data.  
• Experiment with different models (OpenAI GPT, HuggingFace, Ollama) via LangChain wrappers.  
• Interactively explore data and LLM outputs in a Streamlit web app.  
• Keep environment consistent using Docker with GPU support.  

You need:  
• API keys (e.g., `OPENAI_API_KEY`, `HF_TOKEN`) to call paid/free LLM services.  
• Local LLM setup if you use Ollama or certain HuggingFace models.  
• Docker (and NVIDIA Container Toolkit if you want GPU inside containers).


### 📂 Folders and Files

🛠️ **Dockerfile**  
Creates a Docker image with all required dependencies, libraries, tools, and repositories—ensuring consistency and reproducibility.

🧱 **docker-compose.yml** *(optional)*  
Defines services for Jupyter Notebook, Ollama server, and the Streamlit app, with volume mounts and GPU access.

📓 **Notebooks folder**  
Contains Jupyter notebooks that demonstrate how to load data, clean it, call LLMs, and visualise results.

🌐 **Streamlit application folder**  
Includes all the files necessary to run the Streamlit web application. This includes Python scripts, HTML templates, CSS files, and other resources for both frontend and backend functionality.

🧠 **src folder**  
Holds reusable Python modules and packages for the project. These modules centralise data loading and processing, LLM integration logic, and shared helper functions used across notebooks and the Streamlit app.

🗃️ **Datasets folder**  
Stores datasets used throughout the project for data analysis, model training, and algorithm testing. It serves as a central data repository supporting experimentation and validation across various domains.


### Notebooks:
• **Codegemma_x_LangChain.ipynb**: Code snippets for running the 'codegemma' model based on Ollama within the LangChain framework.

• **Llama2_x_LangChain.ipynb**: Utilizes the 'llama2' model within the LangChain framework for NLP tasks.

• **Llama3_x_LangChain.ipynb**: Showcases 'llama3' model capabilities for text analysis and generation.

• **LLaVa_x_LangChain.ipynb**: Integrates 'llava' model into the LangChain framework for text processing.

• **Mistral_x_LangChain.ipynb**: Demonstrates 'mistral' model usage for NLP tasks within the LangChain framework.

• **Mixtral_x_LangChain.ipynb**: Highlights 'mixtral' model functionalities for various text-related tasks.

• **OpenAI_GPT_x_LangChain.ipynb**: Utilizes 'gpt-4' model within the LangChain framework for NLP tasks.

• **Snip.ipynb**: Contains the latest code snippets, functions, and techniques used in the project, including optimizations and GPU utilization.

These folders and files collectively contribute to the development, experimentation, and documentation of the preliminary analysis of datasets using LLMs. 
They serve as valuable resources for understanding, implementing, and extending the project's functionalities.

![File Formats](https://raw.githubusercontent.com/Vibhu249/-Preliminary-Analysis-of-Datasets-Using-LLMs/main/Fileformats.png)


![Performance Heatmap](https://raw.githubusercontent.com/Vibhu249/-Preliminary-Analysis-of-Datasets-Using-LLMs/main/Performance_heatmap.png)


### Installing Local Models:
To install and use the local models, follow these steps:

• **Install Unix/Linux on Windows (if using)**. Alternatively, you can directly run ollama setup in Linux and Mac.

• **Download Ollama setup for Linux**.

• **Open CMD prompt and run the command** ‘ollama pull’ to pull a model from the repository.

• Once the model is successfully installed, run ‘ollama serve’ to start its server.

• To run, simply open any of the Ollama-based Jupyter Notebook files and execute the code. It will automatically start generating results.

