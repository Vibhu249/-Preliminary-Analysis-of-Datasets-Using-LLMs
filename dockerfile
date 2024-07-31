# Use TensorFlow GPU-enabled Jupyter image as the base
FROM tensorflow/tensorflow:latest-gpu-jupyter

# For passing sensitive info like API keys(environment variable)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Install necessary packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio transformers pandas voila \
    ipyvuetify ipywidgets requests openai sentencepiece geopandas rasterio pyproj seaborn \
    scikit-learn xgboost keras langchain_experimental tabulate pandasai langchain langchain-community \
    numpy matplotlib tabula-py jpype1 plotly gpustat statsmodels

# Verify CUDA and PyTorch are working together
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install Java and Git
RUN apt-get update && \
    apt-get install -y default-jdk git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

# Set the working directory
WORKDIR /app

# Clone the external Ollama repository into the container
RUN git clone https://github.com/jmorganca/ollama.git

# Change the working directory to the ollama directory
WORKDIR /app/ollama

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the latest Ollama package is installed
RUN pip uninstall -y ollama && \
    pip install --no-cache-dir ollama

# Expose necessary ports
EXPOSE 8888
EXPOSE 11434

# Start Jupyter Notebook
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
