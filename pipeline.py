import pandas as pd
import numpy as np
import os
import logging
import sys
import importlib
import wandb # NEW: Import wandb
from sklearn.model_selection import train_test_split
# Reload custom modules to ensure the latest changes are picked up
import scripts.data_modelling
importlib.reload(scripts.data_modelling)
import scripts.data_evaluate
importlib.reload(scripts.data_evaluate)
import scripts.model_tuning
importlib.reload(scripts.model_tuning)
import scripts.model_explainability # NEW: Import model_explainability for reload
importlib.reload(scripts.model_explainability) # NEW: Reload model_explainability

import tensorflow as tf
from sklearn.linear_model import LogisticRegression

# Custom Scripts
from scripts.data_loading import DataDownload
from scripts.data_profiling import DataProfiler
from scripts.data_visualizer import DataVisualizer
from scripts.data_preprocessor import DataPreprocessor
from scripts.data_modelling import DataModeling
from scripts.data_evaluate import ModelEvaluator
from scripts.model_tuning import ModelTuner
from scripts.model_explainability import ModelExplainer


# --- Configure Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('pipeline.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def is_running_in_colab():
    """Checks if the code is running in Google Colab."""
    return 'google.colab' in sys.modules

if __name__ == "__main__":
    is_colab = is_running_in_colab()
    logger.info(f"Running in Google Colab: {is_colab}")

    # Define common configuration for WandB and pipeline
    config = {
        "project_name": "flight-delay-prediction",
        "run_name": "full_pipeline_run_with_explainability",
        "dataset_name": "patrickzel/flight-delay-and-cancellation-data-2019-2023-v2",
        "data_sample_file": "flights_sample_10k.csv",
        "target_column": "FLIGHT_STATUS",
        "arr_delay_threshold_mins": 15, # For binary classification target
        "test_size": 0.15,
        "validation_split_ratio": 0.15 / (1 - 0.15), # ~0.176 of train_val for val
        "random_state": 42,
        "high_correlation_threshold": 0.9,
        "eda_top_n_destinations": 20,
        "eda_min_flight_count_airline_delay": 500,
        "shap_explain_sample_size": 1000,
        "nn_epochs": 5, # Example from test_nn_shap_explainability.py
        "nn_batch_size": 32 # Example from test_nn_shap_explainability.py
    }

    # Initialize WandB Run
    try:
        wandb_run = wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config # Log configuration
        )
        logger.info(f"WandB run initialized: {wandb_run.url}")
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}. Proceeding without WandB logging.", exc_info=True)
        wandb_run = None # Set to None if initialization fails


    # Adjust base directory for Colab if needed
    if is_colab:
        base_dir = '/content/drive/MyDrive/Colab Notebooks/flight_delay' # A consistent base directory
        os.makedirs(base_dir, exist_ok=True)
        os.chdir(base_dir)
        logger.info(f"Changed current working directory to {os.getcwd()}")

        try:
            import kaggle
        except ImportError:
            logger.info("Kaggle library not found, installing...")
            os.system("pip install kaggle")
            logger.info("Kaggle library installed.")

    # Define paths relative to the current working directory
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    reports_dir = "reports"
    eda_plots_dir = "plots/eda"
    model_eval_plots_dir = "plots/model_evaluation"
    models_dir = "models"
    model_params_dir = "model_params"
    explainability_plots_dir = "plots/explainability"

    logger.info("--- Starting ML Project Pipeline ---")
    preprocessor = DataPreprocessor()
    # NEW: Pass wandb_run to ModelEvaluator and ModelExplainer
    evaluator = ModelEvaluator(output_dir=models_dir, plots_dir=model_eval_plots_dir, wandb_run=wandb_run)
    tuner = ModelTuner(output_dir=model_params_dir) # ModelTuner does not directly log plots/metrics here
    modeler = DataModeling(tuner=tuner)
    explainer = ModelExplainer(output_dir=explainability_plots_dir, wandb_run=wandb_run)


    # --- Step 1: Data Download ---
    logger.info("\n--- Step 1: Data Download ---")
    try:
        downloader = DataDownload(dataset_name=config["dataset_name"],
                                  download_path=raw_data_dir)
        dataset_directory = downloader.data_download()
        logger.info(f"Dataset download/check complete. Data should be accessible via: {dataset_directory}")
        # NEW: Log raw data path as artifact
        if wandb_run:
            artifact = wandb.Artifact(name="raw-flights-data", type="dataset")
            artifact.add_dir(raw_data_dir) # Add the whole directory
            wandb_run.log_artifact(artifact)
            logger.info("Raw data artifact logged to WandB.")

    except Exception as e:
        logger.critical(f"Critical error during data download: {e}", exc_info=True)
        if wandb_run:
            wandb_run.log_code(".") # Log code on failure
            wandb_run.finish(exit_code=1)
        exit(1)

    # --- Step 2: Data Loading & Initial Filtering ---
    logger.info("\n--- Step 2: Data Loading & Initial Filtering ---")
    csv_file_path = os.path.join(raw_data_dir, config["data_sample_file"])
    df_raw = None

    logger.info(f"Attempting to load CSV from: {csv_file_path}")
    try:
        df_raw = pd.read_csv(csv_file_path)
        logger.info("Dataset loaded successfully!")
        logger.info(f"First 5 rows of the raw dataframe:\n{df_raw.head().to_string()}")
        logger.info(f"Shape of the raw dataframe: {df_raw.shape}")

        initial_shape_before_filter = df_raw.shape
        df_filtered_cancelled = df_raw[df_raw['CANCELLED'] == 0].copy()
        logger.info(f"Filtered for non-cancelled flights. Original shape: {initial_shape_before_filter}, Filtered shape: {df_filtered_cancelled.shape}")

    except FileNotFoundError:
        logger.critical(f"Error: The expected CSV file '{csv_file_path}' was not found. Cannot proceed.", exc_info=True)
        if wandb_run:
            wandb_run.log_code(".")
            wandb_run.finish(exit_code=1)
        exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        if wandb_run:
            wandb_run.log_code(".")
            wandb_run.finish(exit_code=1)
        exit(1)


    # --- Step 3: Data Profiling (Pre-preprocessing) ---
    if df_filtered_cancelled is not None:
        logger.info("\n--- Step 3: Data Profiling (Pre-preprocessing) ---")
        report_path = os.path.join(reports_dir, "flight_data_profile_report_pre_processing.html")
        if os.path.exists(report_path):
            logger.info(f"Profiling report already exists at: {report_path}. Skipping generation.")
            if wandb_run:
                # Log existing report as artifact if it exists and wasn't logged before
                artifact = wandb.Artifact(name="pre_processing_data_profile", type="report")
                artifact.add_file(report_path)
                wandb_run.log_artifact(artifact)
                logger.info("Existing profiling report artifact logged to WandB.")
        else:
            try:
                profiler = DataProfiler(output_dir=reports_dir)
                report_path = profiler.generate_profile_report(df_filtered_cancelled, report_name="flight_data_profile_report_pre_processing.html")
                logger.info(f"Pre-processing data profiling complete. Report saved to: {report_path}")
                # NEW: Log profiling report as artifact
                if wandb_run:
                    artifact = wandb.Artifact(name="pre_processing_data_profile", type="report")
                    artifact.add_file(report_path)
                    wandb_run.log_artifact(artifact)
                    logger.info("New profiling report artifact logged to WandB.")
            except Exception as e:
                logger.error(f"Error during pre-processing data profiling: {e}", exc_info=True)
    else:
        logger.warning("\n--- Data Profiling Skipped (DataFrame is None) ---")

    # --- Step 4: Data Preprocessing for Visualization (EDA) ---
    df_eda = None
    if df_filtered_cancelled is not None:
        logger.info("\n--- Step 4: Data Preprocessing for Visualization (EDA) ---")
        df_eda = df_filtered_cancelled.copy()

        try:
            if 'ARR_DELAY' in df_eda.columns:
                status = []
                for value in df_eda['ARR_DELAY']:
                    status.append(0 if value <= 10 else 1)
                df_eda['FLIGHT_STATUS_EDA'] = status
                logger.info(f"Added 'FLIGHT_STATUS_EDA' column for EDA. Value counts:\n{df_eda['FLIGHT_STATUS_EDA'].value_counts().to_string()}")
            else:
                logger.warning("'ARR_DELAY' column not found in EDA DataFrame for 'FLIGHT_STATUS_EDA' creation.")

            if 'FL_DATE' in df_eda.columns:
                df_eda['YEAR'] = pd.to_datetime(df_eda['FL_DATE']).dt.year
                df_eda['MONTH'] = pd.to_datetime(df_eda['FL_DATE']).dt.month
                logger.info("Added 'YEAR' and 'MONTH' columns for EDA time-series plots.")

            df_eda = preprocessor.create_elapsed_time_diff(df_eda)

            logger.info(f"EDA preprocessing complete. EDA DataFrame shape: {df_eda.shape}")

        except Exception as e:
            logger.error(f"Error during EDA data preprocessing: {e}", exc_info=True)
            df_eda = None

    else:
        logger.warning("\n--- EDA Data Preprocessing Skipped (Data not loaded) ---")


    # --- Step 5: Data Visualization (EDA) ---
    if df_eda is not None:
        logger.info("\n--- Step 5: Data Visualization (EDA) ---")
        try:
            visualizer = DataVisualizer(output_dir=eda_plots_dir)

            # Generate and log EDA plots
            visualizer.plot_column_distribution(df_eda, n_graph_shown=15, n_graph_per_row=4, filename="all_column_distributions_eda.png")
            if wandb_run: wandb_run.log({"EDA/column_distributions": wandb.Image(os.path.join(eda_plots_dir, "all_column_distributions_eda.png"))})

            visualizer.plot_airline_counts(df_eda, filename="airline_flight_counts_eda.png")
            if wandb_run: wandb_run.log({"EDA/airline_flight_counts": wandb.Image(os.path.join(eda_plots_dir, "airline_flight_counts_eda.png"))})

            visualizer.plot_destination_visits(df_eda, top_n=config["eda_top_n_destinations"], filename="top_20_destination_visits_eda.png")
            if wandb_run: wandb_run.log({"EDA/top_destination_visits": wandb.Image(os.path.join(eda_plots_dir, "top_20_destination_visits_eda.png"))})

            visualizer.plot_average_arrival_delay_by_airline(df_eda, min_flight_count=config["eda_min_flight_count_airline_delay"], filename="avg_arrival_delay_by_airline_eda.png")
            if wandb_run: wandb_run.log({"EDA/avg_arrival_delay_by_airline": wandb.Image(os.path.join(eda_plots_dir, "avg_arrival_delay_by_airline_eda.png"))})

            visualizer.plot_total_delays_by_year(df_eda, filename="total_delays_by_year_eda.png")
            if wandb_run: wandb_run.log({"EDA/total_delays_by_year": wandb.Image(os.path.join(eda_plots_dir, "total_delays_by_year_eda.png"))})

            visualizer.plot_monthly_delays_by_year(df_eda, filename="monthly_delays_by_year_eda.png")
            if wandb_run: wandb_run.log({"EDA/monthly_delays_by_year": wandb.Image(os.path.join(eda_plots_dir, "monthly_delays_by_year_eda.png"))})

            visualizer.plot_monthly_trend_with_highlight(df_eda, 'ARR_DELAY', 'Monthly Total Delays Over Time', 'Total Delays (minutes)', filename="monthly_delay_trend_highlight_eda.png")
            if wandb_run: wandb_run.log({"EDA/monthly_delay_trend": wandb.Image(os.path.join(eda_plots_dir, "monthly_delay_trend_highlight_eda.png"))})

            visualizer.plot_delay_reason_analysis(df_eda, filename="delay_reason_breakdown_eda.png")
            if wandb_run: wandb_run.log({"EDA/delay_reason_breakdown": wandb.Image(os.path.join(eda_plots_dir, "delay_reason_breakdown_eda.png"))})

            logger.info("Data visualization complete. Plots saved to 'plots/eda/' directory and logged to WandB.")
        except Exception as e:
            logger.error(f"Error during data visualization: {e}", exc_info=True)
    else:
        logger.warning("\n--- Data Visualization Skipped (EDA DataFrame is None) ---")


    # --- Step 6: Data Preprocessing for Modeling ---
    df_model = None
    if df_filtered_cancelled is not None:
        logger.info("\n--- Step 6: Data Preprocessing for Modeling ---")
        df_model = df_filtered_cancelled.copy()

        try:
            # --- Convert to Classification Target ---
            if 'ARR_DELAY' in df_model.columns:
                df_model[config["target_column"]] = (df_model['ARR_DELAY'] > config["arr_delay_threshold_mins"]).astype(int)
                logger.info(f"Created '{config['target_column']}' binary target (1 if ARR_DELAY > {config['arr_delay_threshold_mins']}, 0 otherwise).")
                logger.info(f"{config['target_column']} value counts:\n{df_model[config['target_column']].value_counts().to_string()}")
            else:
                logger.critical("'ARR_DELAY' column not found for creating classification target.")
                if wandb_run:
                    wandb_run.log_code(".")
                    wandb_run.finish(exit_code=1)
                exit(1)

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
                'FLIGHT_STATUS_EDA',
                'CRS_DEP_TIME', 'CRS_ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'DEP_TIME', 'ARR_TIME',
                'FL_DATE', 'ORIGIN', 'DEST',
                'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
                'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT',
                'ARR_DELAY'
            ]
            df_model = preprocessor.exclude_columns(df_model, columns_to_exclude_model)

            df_model = preprocessor.encode_categorical_features(df_model)

            df_for_corr_check = df_model.drop(columns=[config["target_column"]], errors='ignore')
            columns_to_drop_high_corr, _ = preprocessor.identify_high_correlation_pairs(df_for_corr_check, threshold=config["high_correlation_threshold"])

            if columns_to_drop_high_corr:
                logger.info(f"Removing {len(columns_to_drop_high_corr)} columns due to high correlation: {columns_to_drop_high_corr}")
                df_model = df_model.drop(columns=list(columns_to_drop_high_corr), errors='ignore')
            else:
                logger.info("No columns identified for removal due to high correlation (threshold > 0.9).")

            logger.info("Data preprocessing for modeling complete.")
            logger.info(f"Final Modeling DataFrame shape: {df_model.shape}")
            logger.info(f"Final Modeling DataFrame columns (first 10):\n{df_model.columns.tolist()[:10]}...")
            logger.info(f"Final Modeling DataFrame columns (last 10):\n{df_model.columns.tolist()[-10:]}")

            output_filepath_model = os.path.join(processed_data_dir, 'preprocessed_flight_data_for_modeling.csv')
            df_model.to_csv(output_filepath_model, index=False)
            logger.info(f"Preprocessed data for modeling saved to: {output_filepath_model}")
            # NEW: Log preprocessed data as artifact
            if wandb_run:
                artifact = wandb.Artifact(name="preprocessed-modeling-data", type="processed_data")
                artifact.add_file(output_filepath_model)
                wandb_run.log_artifact(artifact)
                logger.info("Preprocessed data artifact logged to WandB.")


        except Exception as e:
            logger.critical(f"Critical error during data preprocessing for modeling: {e}", exc_info=True)
            df_model = None
            if wandb_run:
                wandb_run.log_code(".")
                wandb_run.finish(exit_code=1)
            exit(1)
    else:
        logger.warning("\n--- Modeling Data Preprocessing Skipped (Data not loaded) ---")

    # --- Step 7: Data Modeling and Evaluation ---
    if df_model is not None:
        logger.info("\n--- Step 7: Data Modeling and Evaluation ---")

        if config["target_column"] not in df_model.columns:
            logger.critical(f"Target column '{config['target_column']}' not found in the modeling DataFrame. Cannot proceed with modeling.")
            if wandb_run:
                wandb_run.log_code(".")
                wandb_run.finish(exit_code=1)
            exit(1)

        X = df_model.drop(columns=[config["target_column"]])
        y = df_model[config["target_column"]]

        logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        logger.info(f"Target variable distribution:\n{y.value_counts()}")

        initial_X_rows = X.shape[0]
        X = X.dropna(axis=1, how='all')
        rows_before_final_nan_drop = X.shape[0]
        X = X.dropna()
        y = y[X.index]

        if X.shape[0] < rows_before_final_nan_drop:
            rows_removed_nan = rows_before_final_nan_drop - X.shape[0]
            logger.warning(f"Removed {rows_removed_nan} rows due to NaN values in features after final preprocessing.")
            logger.info(f"Updated Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")


        if X.empty or y.empty:
            logger.critical("Features or target DataFrame is empty after NaN handling. Cannot proceed with modeling.")
            if wandb_run:
                wandb_run.log_code(".")
                wandb_run.finish(exit_code=1)
            exit(1)

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()

        cols_to_scale = []
        for col in numerical_features:
            is_binary = X[col].dropna().isin([0, 1]).all()
            if not is_binary and not (col.endswith('_SIN') or col.endswith('_COS')):
                cols_to_scale.append(col)

        scaler_name = 'feature_scaler_classification'
        scaler = evaluator.load_model(scaler_name)

        if scaler:
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])
            logger.info("Loaded and applied existing StandardScaler to numerical features.")
        else:
            from sklearn.preprocessing import StandardScaler # Import here if not already
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            evaluator.save_model(scaler, scaler_name)
            logger.info("Trained and saved new StandardScaler for numerical features.")

        # --- Train-Validation-Test Split ---
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=config["validation_split_ratio"], random_state=config["random_state"], stratify=y_train_val)

        logger.info(f"Data split: Train-Val (for HPO/Training)={X_train_val.shape[0]} samples, Test (final evaluation)={X_test.shape[0]} samples")
        logger.info(f"Further split of Train-Val: Train={X_train.shape[0]} samples, Validation={X_val.shape[0]} samples")
        logger.info(f"Test Set Target Distribution:\n{y_test.value_counts()}")

        models_config = {
            "baseline_model": {"func": modeler.run_baseline_model_classification, "train_args": (y_train_val, y_test), "predict_data": y_test},
            "logistic_regression_model": {"func": modeler.run_logistic_regression, "train_args": (X_train_val, y_train_val), "predict_data": X_test},
            "neural_network_classification_model": {"func": modeler.run_neural_network_classification,
                                                    "train_args": (X_train, y_train, X_val, y_val, config["nn_epochs"], config["nn_batch_size"]), # Pass NN params
                                                    "predict_data": X_test}
        }

        trained_models = {}
        for model_name, cfg in models_config.items():
            logger.info(f"\n--- Processing {model_name.replace('_', ' ').title()} ---")

            model = evaluator.load_model(model_name)

            if model is not None:
                logger.info(f"Loaded existing model: {model_name}")
            else:
                logger.info(f"Training new model: {model_name}")
                model = cfg["func"](*cfg["train_args"])
                evaluator.save_model(model, model_name)

            trained_models[model_name] = model

            y_pred_proba = None
            y_pred = None

            if model_name == "baseline_model":
                majority_class = y_train_val.mode()[0]
                y_pred_proba = np.full(len(cfg["predict_data"]), majority_class).astype(float) # Ensure float for proba
                y_pred = y_pred_proba.astype(int)
            elif isinstance(model, tf.keras.Model):
                y_pred_proba = model.predict(cfg["predict_data"]).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(cfg["predict_data"])[:, 1]
                y_pred = model.predict(cfg["predict_data"])

            evaluator.evaluate_classification_model(
                model_name.replace('_', ' ').title(),
                y_test, y_pred, y_pred_proba, stage="Test"
            )

            if model_name != "baseline_model":
                evaluator.plot_roc_curve(model_name.replace('_', ' ').title(), y_test, y_pred_proba, stage="Test")
                evaluator.plot_confusion_matrix(model_name.replace('_', ' ').title(), y_test, y_pred, stage="Test")

            # --- Step 8: Model Explainability ---
            if (model_name == "neural_network_classification_model" or
                model_name == "logistic_regression_model") and trained_models[model_name] is not None:
                logger.info(f"\n--- Step 8: Model Explainability ({model_name.replace('_', ' ').title()}) ---")
                try:
                    X_test_sample_for_shap = X_test.sample(n=min(config["shap_explain_sample_size"], X_test.shape[0]), random_state=config["random_state"])
                    task_type = 'classification'

                    explainer.plot_shap_summary(
                        model=trained_models[model_name],
                        X=X_test_sample_for_shap,
                        filename=f"{model_name}_shap_summary_plot.png",
                        task_type=task_type
                    )

                    potential_top_features = X_test_sample_for_shap.columns.tolist()

                    features_for_dependence = []
                    # Example features - adjust based on your actual data/importance
                    if 'CRS_ELAPSED_TIME' in potential_top_features: features_for_dependence.append('CRS_ELAPSED_TIME')
                    if 'DISTANCE' in potential_top_features: features_for_dependence.append('DISTANCE')
                    if 'AIRLINE_WN' in potential_top_features: features_for_dependence.append('AIRLINE_WN')
                    if 'ORIGIN_STATE_CA' in potential_top_features: features_for_dependence.append('ORIGIN_STATE_CA')
                    if 'DAY_OF_WEEK_is_weekend' in potential_top_features: features_for_dependence.append('DAY_OF_WEEK_is_weekend')
                    if 'CRS_DEP_TIME_SIN' in potential_top_features: features_for_dependence.append('CRS_DEP_TIME_SIN')


                    if features_for_dependence:
                        for feature in features_for_dependence:
                            if feature in X_test_sample_for_shap.columns: # Double-check after sampling
                                explainer.plot_shap_dependence(
                                    model=trained_models[model_name],
                                    X=X_test_sample_for_shap,
                                    feature=feature,
                                    filename=f"{model_name}_shap_dependence_plot_{feature}.png",
                                    task_type=task_type
                                )
                            else:
                                logger.warning(f"Feature '{feature}' not found in X_test_sample_for_shap. Skipping dependence plot.")
                    else:
                        logger.info(f"No suitable features found for SHAP dependence plots for {model_name}.")

                    if model_name == "logistic_regression_model":
                        explainer.plot_coefficient_importance(
                            model=trained_models[model_name],
                            feature_names=X_test_sample_for_shap.columns.tolist(), # Use feature names from the sampled data
                            filename=f"{model_name}_coefficient_importance.png"
                        )
                        logger.info(f"Coefficient importance plot generated for {model_name.replace('_', ' ').title()}.")

                except Exception as e:
                    logger.error(f"Error during model explainability for {model_name.replace('_', ' ').title()}: {e}", exc_info=True)
            else:
                logger.info("Model explainability skipped for other models or if model not trained.")

        evaluator.display_results_table()

    else:
        logger.warning("\n--- Data Modeling Skipped (Modeling DataFrame is None) ---")

    logger.info("\n--- ML Project Pipeline Complete ---")
    if wandb_run:
        wandb_run.log_code(".") # Log all code in the current directory at the end
        wandb_run.finish() # End the WandB run
