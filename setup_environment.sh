#!/bin/bash

# Script to setup virtual environment and install required packages for StockAnalysis

# Set the Python version (change this if you want to use a different version)

# Define possible Python executables
PYTHON_EXECUTABLES=("python3.12" "python3")

# Check if any of the Python executables are available
PYTHON_FOUND=false
for PYTHON in "${PYTHON_EXECUTABLES[@]}"; do
    if command -v $PYTHON &> /dev/null; then
        PYTHON_FOUND=true
        PYTHON_VERSION=$PYTHON
        echo "Python executable found: $PYTHON"
        break
    fi
done

if [ "$PYTHON_FOUND" = false ]; then
    echo "No suitable Python executable found. Please install Python 3.11 or a compatible version."
    exit 1
fi


# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_VERSION -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source $(pwd)/.venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_VERSION -m pip install --upgrade pip

# Install required packages from requirements.txt
echo "Installing required packages..."
$PYTHON_VERSION -m pip install -r requirements.txt

# Set up IPython kernel for Jupyter
echo "Setting up IPython kernel for Jupyter..."
python -m ipykernel install --user --name=stockanalysis

echo "Setup complete. Virtual environment is activated and packages are installed."
echo "To deactivate the virtual environment, run 'deactivate'"
echo "To activate it again later, run 'source .venv/bin/activate'"
echo "To start Jupyter Notebook, run 'jupyter notebook'"