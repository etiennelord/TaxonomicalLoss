#!/bin/bash

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed, proceeding with conda environment setup."
    ENV_NAME="taxonomic_learning"

    # Create conda environment
    conda create -n $ENV_NAME python=3.9 -y
    source activate $ENV_NAME

    # Install dependencies
    conda install -y numpy pandas matplotlib scikit-learn tqdm pillow biopython networkx cudatoolkit cudnn
    conda install -y -c conda-forge timm

    # Check for CUDA GPU
    if nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected. Installing PyTorch with CUDA support."
        #conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        pip install torch torchvision torchaudio
    else
        echo "No CUDA GPU detected. Installing CPU-only version of PyTorch."
        #conda install pytorch torchvision torchaudio cpuonly -c pytorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    pip install vit-pytorch

    echo "Conda environment '$ENV_NAME' is set up and ready to use."
    echo "To activate the environment, run: conda activate $ENV_NAME"

else
    echo "Conda not found. Proceeding with pip installation."

    # Create a virtual environment
    python3 -m venv taxonomic_learning_env
    source taxonomic_learning_env/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    pip install numpy pandas matplotlib scikit-learn tqdm pillow biopython networkx timm

    # Check for CUDA GPU
    if nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected. Installing PyTorch with CUDA support."
        pip install torch torchvision torchaudio
    else
        echo "No CUDA GPU detected. Installing CPU-only version of PyTorch."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    pip install vit-pytorch

    echo "Virtual environment is set up and ready to use."
    echo "To activate the environment, run: source taxonomic_learning_env/bin/activate"
fi

echo "Installation complete. You can now run your Python scripts."
