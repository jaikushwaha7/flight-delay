import pandas as pd
import numpy as np
import os
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import importlib

# Import your custom scripts
from scripts.data_loading import DataDownload
from scripts.data_preprocessor import DataPreprocessor
from scripts.model_explainability import ModelExplainer
from scripts.data_evaluate import ModelEvaluator # To load scaler and model

# --- Configure Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Define Paths ---
raw_data_dir = "data/raw"
processed_data_dir = "data/processed" # Ensure this exists if you need to save preprocessed data
models_dir = "models"
explainability_plots_dir = "plots/explainability"

nn_model_filename = "neural_network_classification_model.h5"
nn_model_path = os.path.join(models_dir, nn_model_filename)
scaler_filename = "feature_scaler_classification" # Name of your saved scaler
scaler_path = os.path.join(models_dir, f"{scaler_filename}.pkl") # Assuming .pkl extension for scaler

# Ensure directories exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(explainability_plots_dir, exist_ok=True)


if __name__ == "__main__":
    logger.info("--- Starting SHAP Explainability Test for Neural Network with Flights Data ---")

    # Initialize necessary components from your pipeline
    downloader = DataDownload(dataset_name="patrickzel/flight-delay-and-cancellation-data-2019-2023-v2",
                              download_path=raw_data_dir)
    preprocessor = DataPreprocessor()
    evaluator = ModelEvaluator(output_dir=models_dir, plots_dir="plots/model_evaluation") # Need evaluator to load model/scaler
    explainer = ModelExplainer(output_dir=explainability_plots_dir)

    # --- 1. Data Download & Initial Filtering (mimic pipeline.py) ---
    logger.info("\n--- Step 1: Data Download ---")
    try:
        downloader.data_download()
        csv_file_path = os.path.join(raw_data_dir, 'flights_sample_10k.csv')
        df_raw = pd.read_csv(csv_file_path)
        df_filtered_cancelled = df_raw[df_raw['CANCELLED'] == 0].copy()
        logger.info(f"Loaded and filtered flights data. Shape: {df_filtered_cancelled.shape}")
    except Exception as e:
        logger.critical(f"Error during data loading: {e}", exc_info=True)
        exit(1)

    # --- 2. Data Preprocessing for Modeling (mimic pipeline.py) ---
    logger.info("\n--- Step 2: Data Preprocessing for Modeling ---")
    df_model = df_filtered_cancelled.copy()

    try:
        # --- Convert to Classification Target ---
        df_model['FLIGHT_STATUS'] = (df_model['ARR_DELAY'] > 15).astype(int)

        delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
                      'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                      'DELAY_DUE_LATE_AIRCRAFT']
        df_model = preprocessor.handle_missing_values(df_model, delay_cols)

        df_model = preprocessor.create_elapsed_time_diff(df_model)

        time_columns = ['CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME']
        df_model = preprocessor.apply_cyclical_encoding(df_model, time_columns)

        city_state_columns = ['DEST_CITY', 'ORIGIN_CITY']
        df_model = preprocessor.split_city_state(df_model, city_state_columns)

        date_columns = ['FL_DATE']
        df_model = preprocessor.add_weekday_weekend_columns(df_model, date_columns)

        columns_to_exclude_model = [
            'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
            'CANCELLED', 'DIVERTED', 'CANCELLATION_CODE',
            'FLIGHT_STATUS_EDA', # Drop EDA specific column from pipeline's EDA step
            'CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME',
            'FL_DATE', 'ORIGIN', 'DEST',
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
            'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
            'ARR_DELAY'
        ]
        df_model = preprocessor.exclude_columns(df_model, columns_to_exclude_model)

        df_model = preprocessor.encode_categorical_features(df_model)

        # High Correlation Feature Removal (re-evaluate this with your actual features if necessary)
        df_for_corr_check = df_model.drop(columns=['FLIGHT_STATUS'], errors='ignore')
        columns_to_drop_high_corr, _ = preprocessor.identify_high_correlation_pairs(df_for_corr_check, threshold=0.9)
        if columns_to_drop_high_corr:
            df_model = df_model.drop(columns=list(columns_to_drop_high_corr), errors='ignore')
            logger.info(f"Removed {len(columns_to_drop_high_corr)} columns due to high correlation during preprocessing for modeling.")


        logger.info(f"Preprocessing for modeling complete. Final DataFrame shape: {df_model.shape}")
        # Make sure all columns that were available during training are present,
        # and any new columns are handled (e.g., if one-hot encoding creates more)
        # It's crucial that X.columns match exactly what the model was trained on.

    except Exception as e:
        logger.critical(f"Critical error during data preprocessing for modeling: {e}", exc_info=True)
        exit(1)

    # --- 3. Prepare Data for Model (mimic pipeline.py train/test split and scaling) ---
    logger.info("\n--- Step 3: Preparing Data for Model ---")
    if 'FLIGHT_STATUS' not in df_model.columns:
        logger.critical("Target column 'FLIGHT_STATUS' not found. Cannot proceed.")
        exit(1)

    X = df_model.drop(columns=['FLIGHT_STATUS'])
    y = df_model['FLIGHT_STATUS']

    # Final NaN drop from X (as done in pipeline)
    X = X.dropna(axis=1, how='all')
    rows_before_final_nan_drop = X.shape[0]
    X = X.dropna()
    y = y[X.index] # Align y with X
    if X.shape[0] < rows_before_final_nan_drop:
        logger.warning(f"Removed {rows_before_final_nan_drop - X.shape[0]} rows due to NaN values in features during final data prep.")

    # Train-Test Split (use the same random_state and split ratio as in pipeline.py)
    # We only need X_test for explainability
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    logger.info(f"Test set shape before scaling: X_test {X_test.shape}, y_test {y_test.shape}")

    # Load the trained scaler
    scaler = evaluator.load_model(scaler_filename)
    if scaler is None:
        logger.critical(f"Scaler '{scaler_filename}' not found at {scaler_path}. Please run pipeline.py first to train and save the scaler.")
        exit(1)

    # Identify numerical columns that were scaled during pipeline run
    numerical_features_to_scale = [col for col in X_test.select_dtypes(include=np.number).columns if not (X_test[col].dropna().isin([0, 1]).all() or col.endswith('_SIN') or col.endswith('_COS'))]

    # Apply scaling to the test set
    X_test_scaled = X_test.copy()
    if numerical_features_to_scale:
        X_test_scaled[numerical_features_to_scale] = scaler.transform(X_test_scaled[numerical_features_to_scale])
        logger.info(f"Applied StandardScaler to {len(numerical_features_to_scale)} numerical features in X_test.")
    else:
        logger.info("No numerical features to scale in X_test or already handled.")

    # Crucially, ensure X_test_scaled is a DataFrame with original column names
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    logger.info(f"Scaled X_test_df shape: {X_test_scaled_df.shape}")


    # --- 4. Load the Neural Network Model ---
    logger.info("\n--- Step 4: Loading Neural Network Model ---")
    model = evaluator.load_model(nn_model_filename)
    if model is None:
        logger.critical(f"Neural Network model '{nn_model_filename}' not found at {nn_model_path}. Please run pipeline.py first to train and save the model.")
        exit(1)
    logger.info("Neural Network model loaded successfully.")

    # --- 5. Generate SHAP Explainability Plots ---
    logger.info("\n--- Step 5: Generating SHAP Plots ---")
    try:
        # Use a sample of the test data for SHAP to speed up calculation for DeepExplainer
        # DeepExplainer is recommended for Keras models and can handle more samples than KernelExplainer
        # without needing a separate background dataset for its initialization.
        # However, for very large test sets, a sample is still good.
        X_explain_sample = X_test_scaled_df.sample(n=min(1000, X_test_scaled_df.shape[0]), random_state=42)
        logger.info(f"Using {X_explain_sample.shape[0]} samples from scaled test data for SHAP explainability.")

        task_type = 'classification' # Flight delay is a binary classification

        # SHAP Summary Plot
        explainer.plot_shap_summary(
            model=model,
            X=X_explain_sample,
            filename=f"{nn_model_filename.replace('.h5', '')}_shap_summary_plot.png",
            task_type=task_type
        )

        # SHAP Dependence Plots for a few features
        # It's best to identify truly impactful features from the summary plot after its run.
        # For demonstration, let's pick a few features that are often relevant.
        features_for_dependence = []
        if 'CRS_ELAPSED_TIME' in X_explain_sample.columns: features_for_dependence.append('CRS_ELAPSED_TIME')
        if 'DISTANCE' in X_explain_sample.columns: features_for_dependence.append('DISTANCE')
        if 'AIRLINE_WN' in X_explain_sample.columns: features_for_dependence.append('AIRLINE_WN') # Example airline
        if 'ORIGIN_STATE_CA' in X_explain_sample.columns: features_for_dependence.append('ORIGIN_STATE_CA') # Example state
        if 'DAY_OF_WEEK_is_weekend' in X_explain_sample.columns: features_for_dependence.append('DAY_OF_WEEK_is_weekend')
        if 'CRS_DEP_TIME_SIN' in X_explain_sample.columns: features_for_dependence.append('CRS_DEP_TIME_SIN')


        if features_for_dependence:
            for feature in features_for_dependence:
                explainer.plot_shap_dependence(
                    model=model,
                    X=X_explain_sample,
                    feature=feature,
                    filename=f"{nn_model_filename.replace('.h5', '')}_shap_dependence_plot_{feature}.png",
                    task_type=task_type
                )
        else:
            logger.warning("No default features identified for SHAP dependence plots. Consider adding relevant features.")


    except Exception as e:
        logger.error(f"An error occurred during SHAP explainability: {e}", exc_info=True)

    logger.info("--- SHAP Explainability Test Complete ---")
