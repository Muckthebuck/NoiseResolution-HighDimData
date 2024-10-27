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

The full report is available here `Final report_GoupA.pdf`

## Setup Instructions for Luca's code

The code is stored in the file "Formatted code.Rmd". It is a R-Notebook. The code is divided into two sections: Plots/Results and Functions. The first section, Plots/Results, shall be run after the second section "Functions". The order is chosen such that the IDE remains uncluttered.


## Setup Instructions for Mukul's code
### VS Code setup
1. Open then `data_analysis_pipeline.ipynb`.
2. Select Kernel here ![image](https://github.com/user-attachments/assets/694836d8-efdb-4348-9efd-fc276ed1bd4a)
3. Select Python Environments
4. Create Python Environment
5. Venv -> `Python 3.12.7`
6. Follow the prompts on the screen, it will ask you to select the requirements file, select `requirements.txt` and install all the packages required. 

### Requirements file

The following Python packages are required for this project:
- `Python 3.12.7`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `jupyter`
- `notebook`
 - `ipykernel`

If you need to manually install additional packages, you can do so using:
```bash
pip install <package-name>
```

## Data Description

The project uses the following datasets:

- `Stock_metadata.csv`: Metadata table for the stocks.
- `Data_PreCovid_20170101_20200109.csv`: Daily log-returns for each stock during the pre-COVID period (2017-01-01 to 2020-01-09).
- `Data_PostCovid_20200110_20221231.csv`: Daily log-returns for each stock during the post-COVID period (2020-01-10 to 2022-12-31).

Each dataset contains information for 98 stocks from the US market.
