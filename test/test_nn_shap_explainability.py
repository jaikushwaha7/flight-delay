import pandas as pd
import numpy as np
import os
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Assuming ModelExplainer is in scripts/model_explainability.py
from scripts.model_explainability import ModelExplainer

# --- Configure Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def create_dummy_data(n_samples=1000, n_features=20, random_state=42):
    """Generates a dummy classification dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=10,
                               n_redundant=5, n_repeated=2, n_classes=2, random_state=random_state)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name='target')
    logger.info(f"Created dummy dataset: X_df shape {X_df.shape}, y_df shape {y_df.shape}")
    return X_df, y_df

def create_and_save_dummy_nn_model(X_train_scaled, y_train, model_path):
    """Creates, trains, and saves a simple dummy Neural Network model."""
    input_dim = X_train_scaled.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Dummy Neural Network model compiled.")

    # Train for a few epochs
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, verbose=0)
    logger.info("Dummy Neural Network model trained for 5 epochs.")

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Dummy Neural Network model saved to {model_path}")
    return model

if __name__ == "__main__":
    models_dir = "models"
    explainability_plots_dir = "plots/explainability"
    nn_model_filename = "neural_network_classification_model.h5"
    nn_model_path = os.path.join(models_dir, nn_model_filename)

    logger.info("--- Starting SHAP Explainability Test for Neural Network ---")

    # 1. Create Dummy Data
    X, y = create_dummy_data(n_samples=1000, n_features=20)

    # 2. Split and Scale Data (Essential for NN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with column names for SHAP
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    logger.info("Data split and scaled.")

    # 3. Create or Load Dummy Neural Network Model
    if os.path.exists(nn_model_path):
        logger.info(f"Loading existing dummy NN model from {nn_model_path}")
        # Need to compile a dummy model first to load weights correctly
        input_dim = X_train_scaled_df.shape[1]
        dummy_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification
        ])
        dummy_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model = tf.keras.models.load_model(nn_model_path)
        logger.info("Dummy NN model loaded.")
    else:
        logger.info("Creating and saving a new dummy NN model.")
        model = create_and_save_dummy_nn_model(X_train_scaled_df, y_train, nn_model_path)


    # 4. Initialize ModelExplainer
    explainer = ModelExplainer(output_dir=explainability_plots_dir)

    # 5. Generate SHAP Explainability Plots
    logger.info("\n--- Generating SHAP Plots ---")
    try:
        # Use a sample of the test data for SHAP to speed up calculation
        # For KernelExplainer (used by default if DeepExplainer fails or for simple models),
        # a smaller background dataset is needed for initialization.
        # For DeepExplainer, the full X_test_scaled_df can be used for shap_values calculation.
        
        # Taking a smaller sample for explainability for performance.
        # For NN, DeepExplainer is usually faster than KernelExplainer.
        # DeepExplainer can use the full X_test_scaled_df to calculate SHAP values.
        # The background for DeepExplainer is derived from the training data typically.
        
        # To make it work reliably for DeepExplainer, X should be numpy array.
        X_test_sample_for_shap = X_test_scaled_df.sample(n=min(500, X_test_scaled_df.shape[0]), random_state=42)
        logger.info(f"Using {X_test_sample_for_shap.shape[0]} samples for SHAP explainability.")

        # SHAP Summary Plot
        explainer.plot_shap_summary(
            model=model,
            X=X_test_sample_for_shap, # Use a sample for SHAP calculation efficiency
            filename=f"{nn_model_filename.replace('.h5', '')}_shap_summary_plot.png",
            task_type='classification'
        )

        # SHAP Dependence Plots for a few features
        # Choose some features that might be important or interesting from your dataset
        features_to_explain = ['feature_0', 'feature_1', 'feature_2'] # Example features
        for feature in features_to_explain:
            if feature in X_test_sample_for_shap.columns:
                explainer.plot_shap_dependence(
                    model=model,
                    X=X_test_sample_for_shap,
                    feature=feature,
                    filename=f"{nn_model_filename.replace('.h5', '')}_shap_dependence_plot_{feature}.png",
                    task_type='classification'
                )
            else:
                logger.warning(f"Feature '{feature}' not found in the dummy dataset. Skipping dependence plot.")


    except Exception as e:
        logger.error(f"An error occurred during SHAP explainability: {e}", exc_info=True)

    logger.info("--- SHAP Explainability Test Complete ---")
