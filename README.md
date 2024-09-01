# Project Title: Investigation of Methods for Resolving Statistical Noise and Understanding Correlation Structure in High-Dimensional Data

## Description  
This project, conducted as part of the ELEN90094 – Large Data Methods & Applications course at The University of Melbourne, investigates methods for resolving statistical noise and understanding correlation structures in high-dimensional data, with a focus on financial stock return data. The analysis is divided into two stages:

1. **Stage 1**: An in-depth study of data analysis methods for identifying true correlation information in complex, high-dimensional datasets. The study involves reviewing and critiquing approaches presented in the research paper by V. Plerou et al. (Physical Review E, 2002), specifically methods for investigating correlation patterns in financial data.

2. **Stage 2**: Application of the methods studied in Stage 1, along with additional methods covered in the course, to real-world financial stock return datasets from periods before and after the emergence of COVID-19. The analysis aims to distinguish true correlations from noise, quantify statistical noise, and reveal structured correlation patterns.

The project will produce a comprehensive report and code that documents the methods, experiments, and results. The results will provide insights into the dependencies and correlation structures in financial data, and the project will explore the stability and properties of estimated correlation matrices.

**Key Components**:
- Literature review of high-dimensional data analysis methods.
- Application and interpretation of these methods on real-world financial data.
- Documentation of numerical experiments and their results.
- Insights into noise resolution and correlation structures in complex datasets.

## Setup Instructions

To set up the environment and install the required packages for this project, follow these steps:

### Set Up Virtual Environment

1. Clone the repository or navigate to the project directory.
   
2. Run the `setup_environment.sh` script to set up the virtual environment and install the necessary packages:

    ```bash
    ./setup_environment.sh
    ```

This script will:

- Check if Python 3.11 is installed.
- Create a virtual environment named `.venv` in the project directory.
- Activate the virtual environment.
- Upgrade `pip` to the latest version.
- Install the required packages from the `requirements.txt` file.


### Activating and Decativating the Virtual Environment

If the virtual environment is not already activated or you need to activate it again, you can activate it by running:

```bash
source .venv/bin/activate
```

To deactivate the virtual environment, simply run:


```bash
deactivate
```

## Requirements

The following Python packages are required for this project:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `jupyter`
- `notebook`
 - `ipykernel`

These packages will be installed automatically by the `setup_environment.sh` script. If you need to manually install additional packages, you can do so using:
```bash
pip install <package-name>
```

## Data Description

The project uses the following datasets:

- `Stock_metadata.csv`: Metadata table for the stocks.
- `Data_PreCovid_20170101_20200109.csv`: Daily log-returns for each stock during the pre-COVID period (2017-01-01 to 2020-01-09).
- `Data_PostCovid_20200110_20221231.csv`: Daily log-returns for each stock during the post-COVID period (2020-01-10 to 2022-12-31).

Each dataset contains information for 98 stocks from the US market.
