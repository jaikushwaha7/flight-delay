# Predictive Delay Analysis for Airline Operations
## ✈️ Flight Delay Prediction ML Pipeline

## Project Overview

This project implements an end-to-end Machine Learning pipeline for predicting flight delays. The primary goal is to classify whether a flight will experience a significant arrival delay (defined as more than 15 minutes) or be on-time/minorly delayed. The pipeline covers all stages from data ingestion and preprocessing to model training, evaluation, and explainability, leveraging modern MLOps practices like experiment tracking with Weights & Biases (WandB) and an interactive dashboard with Gradio.

**Problem:** Binary Classification (Flight Delayed vs. Flight On-Time/Minor Delay)

## Features

* **Data Ingestion:** Downloads raw flight data from Kaggle.
* **Data Preprocessing:**
    * Handles missing values.
    * Feature engineering (e.g., elapsed time difference, cyclical time features, weekday/weekend indicators, city/state splitting).
    * Categorical feature encoding (One-Hot Encoding).
    * High correlation feature removal.
* **Exploratory Data Analysis (EDA):** Generates various plots to understand data distributions, trends, and relationships.
* **Data Profiling:** Provides a detailed `pandas-profiling` report for data quality assessment.
* **Model Training:**
    * Trains and evaluates multiple classification models:
        * Baseline (Majority Class Predictor)
        * Logistic Regression
        * Neural Network (TensorFlow/Keras)
* **Model Evaluation:** Calculates and visualizes key classification metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC, Confusion Matrix).
* **Model Explainability:** Utilizes SHAP (SHapley Additive exPlanations) to interpret model predictions, showing feature importance and interactions.
* **Experiment Tracking (WandB):** Integrates Weights & Biases for logging metrics, plots, model artifacts, and pipeline configuration, facilitating experiment comparison and reproducibility.
* **Interactive Dashboard (Gradio):** A user-friendly web interface to explore pipeline results, EDA plots, model evaluations, and explainability insights without diving into the code.
* **Modular Design:** The pipeline is structured into independent Python scripts for each stage (data loading, preprocessing, modeling, etc.), promoting maintainability and reusability.

## Project Structure
```

├── data/
│   ├── raw/                      # Raw downloaded data (e.g., flights_sample_100k.csv)
│   └── processed/                # Preprocessed data for modeling
├── models/                       # Trained machine learning models (e.g., .pkl, .h5)
├── model_params/                 # Hyperparameter tuning results or optimal parameters
├── plots/
│   ├── eda/                      # Exploratory Data Analysis plots
│   ├── model_evaluation/         # Model evaluation plots (ROC, Confusion Matrix)
│   └── explainability/           # Model explainability plots (SHAP, Coefficients)
├── reports/                      # Data profiling reports (e.g., HTML)
├── scripts/
│   ├── data_loading.py           # Handles data download from Kaggle
│   ├── data_profiling.py         # Generates data profile reports
│   ├── data_preprocessor.py      # Implements data cleaning and feature engineering
│   ├── data_visualizer.py        # Generates EDA plots
│   ├── data_modelling.py         # Defines and trains ML models
│   ├── data_evaluate.py          # Evaluates models and saves metrics/plots
│   ├── model_tuning.py           # (Optional) For hyperparameter tuning
│   └── model_explainability.py   # Generates SHAP and other explainability plots
├── main.py                       # Orchestrates the entire ML pipeline
├── app.py                        # Gradio application for dashboard
├── config.yaml                   # Configuration file for pipeline parameters
├── requirements.txt              # Project dependencies
├── .env.example                  # Example for environment variables
└── README.md                     # This file
```

## Setup and Installation (Local System)

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* Kaggle API Key (for data download)

### Step 1: Clone the Repository

```bash
git clone <repository_url>
cd flight-delay-prediction-ml-pipeline # Replace with your actual repository name
```
### Step 2: Create a Virtual Environment (Recommended)
It's highly recommended to use a virtual environment to manage project dependencies.

```Bash

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```
### Step 3: Install Dependencies
```Bash
pip install -r requirements.txt
```

### Step 4: Configure Kaggle API
To download the dataset, you need to set up your Kaggle API credentials.
```
Go to your Kaggle account page: https://www.kaggle.com/<your-username>/account

Under the "API" section, click "Create New API Token". This will download a kaggle.json file.

Place this kaggle.json file in the ~/.kaggle/ directory on your system.

Windows: C:\Users\<Windows-username>\.kaggle\

macOS/Linux: ~/.kaggle/
```
Important: Ensure the permissions for kaggle.json are set to 600 (read/write only for owner) to keep your token secure.

```
chmod 600 ~/.kaggle/kaggle.json
```
### Step 5: Configure Environment Variables
The project uses Weights & Biases (WandB) for experiment tracking. You'll need an API key for WandB.

- Sign up for a free WandB account at wandb.ai.

- Find your API key in your WandB settings.

- Create a .env file in the root directory of your project based on .env.example:

```Bash

cp .env.example .env
```
Edit the .env file and replace YOUR_WANDB_API_KEY with your actual WandB API key:

```Ini, TOML

WANDB_API_KEY=YOUR_WANDB_API_KEY
```
The main.py script will automatically load this key.

### Step 6: Review Configuration
Open config.yaml to review or modify pipeline parameters such as dataset name, target thresholds, model hyperparameters, and split ratios.

```YAML

# config.yaml (example snippet)
dataset_name: patrickzel/flight-delays-from-2017-to-2024
data_sample_file: flights_sample_100k.csv
project_name: flight-delay-prediction-final
run_name: full_pipeline_initial_run
target_column: FLIGHT_STATUS
arr_delay_threshold_mins: 15
test_size: 0.2
validation_split_ratio: 0.25 # 0.25 of train_val set (which is 0.25 * 0.8 = 0.2 total data)
random_state: 42
high_correlation_threshold: 0.9
eda_top_n_destinations: 20
eda_min_flight_count_airline_delay: 500
shap_explain_sample_size: 1000 # Number of samples for SHAP explainability
# ... (other model specific parameters)
```
### Usage
There are two main ways to interact with this project:

1. Run the Full ML Pipeline: Executes all steps from data download to model explainability.

2. Launch the Gradio Dashboard: Provides an interactive interface to visualize results from a previously run pipeline.

* Option 1: Running the Full ML Pipeline
This will download data, preprocess it, train models, evaluate them, generate all plots, and log everything to WandB.

```Bash

python pipeline.py
```
* **Output**: You will see logs in your console and a pipeline.log file. All generated data, models, reports, and plots will be saved in their respective directories (data/, models/, plots/, reports/).

    * **WandB**: A link to your WandB run will be printed in the console, where you can explore the experiment results, metrics, and logged artifacts.

* **Option 2**: Launching the Gradio Dashboard
After running main.py at least once to generate the necessary files, you can launch the Gradio dashboard to interactively explore the results.

```Bash

python app.py
```
* **Output**: A local URL (e.g., http://127.0.0.1:7860) and potentially a public shareable Gradio link will be printed in your console.

* **Access**: Open the URL in your web browser. You can navigate through tabs to see:

```
- Basic data information and target definition.
- Exploratory Data Analysis (EDA) plots.
- Data preprocessing summary.
- Links to the full data profiling report.
- Model evaluation metrics and plots (ROC curves, Confusion Matrices).
- Model explainability insights (SHAP summary and dependence plots, Coefficient Importance).
```
**Troubleshooting**
kaggle.json not found or permission issues: Double-check that kaggle.json is in the correct ~/.kaggle/ directory and has 600 permissions.

**Missing Python packages**: Ensure you've activated your virtual environment and run pip install -r requirements.txt.

**WandB API Key**: Verify that WANDB_API_KEY is correctly set in your .env file.

**Plots/Reports not showing in Gradio**: Ensure you have run python pipeline.py successfully at least once. The Gradio app displays pre-generated files. Check the console output of app.py for warnings about missing files.

**Colab Environment**: If running in Google Colab, the pipeline.py script automatically adjusts paths and installs Kaggle. Ensure your Google Drive is mounted correctly if saving results there.