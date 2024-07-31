# Preliminary Analysis of Datasets Using LLMs

https://github.com/Vibhu249/-Preliminary-Analysis-of-Datasets-Using-LLMs/blob/main/FlowChart.png

This repository contains all the necessary files for the preliminary analysis of datasets using large language models (LLMs). 
Below are the details of each folder and file.

### Note:
• You need to have API keys from OpenAI (paid) and HuggingFace (free) to run any GPT models.

• Install the LLMs from Ollama and HuggingFace. Models from Ollama are necessary to run all files, including Python files or the Streamlit web app.

### Folders and Files:
**• Dockerfile**: Creates a Docker image with all required dependencies, libraries, tools, and repositories, ensuring consistency and reproducibility.
#### • Streamlit application folder: This folder contains all the files necessary to run the developed Streamlit Web Application. It includes Python scripts, HTML templates, CSS files, and any other resources required for the application's frontend and backend functionalities.
#### • Dataset Folder: Contains most of the datasets used in the project for analysis, model training, and testing algorithms. It serves as a repository for the data required for conducting analysis, training models, and testing algorithms. The datasets cover a wide range of domains and provide valuable resources for experimentation and validation of the project's methodologies.

### Jupyter Notebooks:
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

### Installing Local Models:
To install and use the local models, follow these steps:

• **Install Unix/Linux on Windows (if using)**. Alternatively, you can directly run ollama setup in Linux and Mac.

• **Download Ollama setup for Linux**.

• **Open CMD prompt and run the command** ‘ollama pull’ to pull a model from the repository.

• Once the model is successfully installed, run ‘ollama serve’ to start its server.

• To run, simply open any of the Ollama-based Jupyter Notebook files and execute the code. It will automatically start generating results.
