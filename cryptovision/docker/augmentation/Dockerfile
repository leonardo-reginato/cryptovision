# Use the official Miniconda3 image as the base
FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    libhdf5-dev \
    build-essential

# Set up environment variables
ENV CONDA_ENV_NAME=cvenv
ENV CONDA_ENV_PYTHON=3.10

# Create the environment with Python 3.10
RUN conda create --name $CONDA_ENV_NAME python=$CONDA_ENV_PYTHON -y

# Activate the environment
RUN echo "source activate $CONDA_ENV_NAME" > ~/.bashrc
ENV PATH /opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH

# Install Jupyter Notebook
RUN conda install jupyter -y

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN /opt/conda/envs/$CONDA_ENV_NAME/bin/pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Notebook port
EXPOSE 8888

# Set the working directory
WORKDIR /CryptoVision

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]