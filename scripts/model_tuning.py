
import logging
import os
import json # To save best parameters
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error, f1_score # Import f1_score for classification

# For Neural Network Tuning
import tensorflow as tf
from tensorflow import keras
import keras_tuner # Make sure this is imported for keras_tuner.HyperModel
from keras_tuner.tuners import RandomSearch, Hyperband

# Import build_nn_model from nn_utils (assuming it's been refactored there)
from scripts.nn_utils import build_nn_model 

# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ModelTuner:
    """
    Handles hyperparameter tuning and cross-validation for machine learning models.
    Saves and loads best parameters to/from a JSON file.
    """
    def __init__(self, output_dir="model_params"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.params_file = os.path.join(self.output_dir, "all_best_params.json")
        self.best_params_cache = self._load_all_best_params() 
        logger.info(f"ModelTuner initialized. Best parameters will be saved to '{self.output_dir}'.")

    def _load_all_best_params(self):
        """Loads all best parameters from the single JSON file."""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding {self.params_file}: {e}. Returning empty cache.", exc_info=True)
                return {}
        return {}

    def _save_all_best_params(self):
        """Saves current best parameters cache to the single JSON file."""
        try:
            with open(self.params_file, 'w') as f:
                json.dump(self.best_params_cache, f, indent=4)
            logger.info(f"All best parameters saved to {self.params_file}")
        except Exception as e:
            logger.error(f"Error saving all best parameters to {self.params_file}: {e}", exc_info=True)

    def get_best_params(self, model_name: str) -> dict:
        """Retrieves best parameters for a specific model from the cache."""
        return self.best_params_cache.get(model_name, None)

    def tune_model(self, model, param_grid: dict, X: pd.DataFrame, y: pd.Series, 
                   model_name: str, cv_splits: int = 3, search_method: str = 'grid', # Reduced cv_splits
                   n_iter: int = 3, scoring: str = 'neg_mean_absolute_error') -> tuple: # Reduced n_iter
        """
        Performs hyperparameter tuning and cross-validation for a given scikit-learn model.
        (Parameters eased out for faster convergence during HPO).

        Args:
            model: The scikit-learn estimator to tune.
            param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter
                                settings to try as values.
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            model_name (str): A descriptive name for the model (e.g., 'ridge_regression').
            cv_splits (int): Number of cross-validation splits. (Reduced for faster HPO)
            search_method (str): 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
            n_iter (int): Number of parameter settings that are sampled if search_method is 'random'. (Reduced for faster HPO)
            scoring (str): Scoring metric to optimize (e.g., 'neg_mean_absolute_error').

        Returns:
            tuple: A tuple containing (best_estimator, best_params, best_score).
        """
        logger.info(f"Starting hyperparameter tuning for {model_name} using {search_method} search...")
        
        best_params_loaded = self.get_best_params(model_name)
        if best_params_loaded:
            logger.info(f"Using loaded best parameters for {model_name}: {best_params_loaded}. Skipping tuning.")
            model.set_params(**best_params_loaded) 
            return model, best_params_loaded, None 

        cv_strategy = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

        if search_method == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv_strategy, scoring=scoring, verbose=0, n_jobs=-1) # Reduced verbose
        elif search_method == 'random':
            search = RandomizedSearchCV(model, param_distro=param_grid, n_iter=n_iter, cv=cv_strategy, scoring=scoring, verbose=0, n_jobs=-1, random_state=42) # Reduced verbose
        else:
            raise ValueError(f"Unknown search_method: {search_method}. Must be 'grid' or 'random'.")

        search.fit(X, y)

        best_estimator = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best CV score for {model_name} ({scoring}): {best_score:.4f}")

        self.best_params_cache[model_name] = best_params
        self._save_all_best_params()
        
        return best_estimator, best_params, best_score

    class NeuralNetworkHyperModel(keras_tuner.HyperModel):
        def __init__(self, input_dim: int, task_type: str = 'regression'):
            self.input_dim = input_dim
            self.task_type = task_type
            super().__init__()

        def build(self, hp):
            """
            Builds a Keras model based on hyperparameters chosen by the tuner.
            (Parameters eased out for faster convergence during HPO).
            """
            # No need to import build_nn_model here if it's imported at the module level.
            # However, if it's explicitly needed here due to circular import issues,
            # ensure it's from the correct place (nn_utils).
            # If `from scripts.nn_utils import build_nn_model` is at the top of this file, remove this local import.
            # If build_nn_model is still in data_modelling.py and there's a circular import, keep this.
            # Given the previous error, it was likely due to a circular import that
            # has now been resolved by suggesting to move build_nn_model to nn_utils.
            # So, assuming build_nn_model is now correctly imported from nn_utils at the top.
            # If not, add: from scripts.nn_utils import build_nn_model here
            
            # Eased out search space for faster tuning
            num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=2, default=1) # Reduced max layers
            neurons = hp.Int('neurons', min_value=32, max_value=64, step=32, default=32) # Reduced range
            dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.1, default=0.1) # Reduced max dropout
            learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4], default=1e-3) # Fewer choices
            
            model = build_nn_model(
                input_dim=self.input_dim,
                hidden_layers=num_hidden_layers,
                neurons=neurons,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                task_type=self.task_type
            )
            return model

    def tune_neural_network_classification(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                           X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Performs hyperparameter tuning for the Neural Network Classification model
        using Keras Tuner. (Parameters eased out for faster convergence during HPO).

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.

        Returns:
            dict: The best hyperparameters found.
        """
        logger.info("\n--- Tuning Neural Network Classification Model ---")

        model_name = "neural_network_classification"
        best_params_loaded = self.get_best_params(model_name)

        if best_params_loaded:
            logger.info(f"Using loaded best parameters for Neural Network Classification: {best_params_loaded}. Skipping tuning.")
            return best_params_loaded
        
        tuner_dir = os.path.join(self.output_dir, "keras_tuner_nn_classification")
        os.makedirs(tuner_dir, exist_ok=True)

        hypermodel = self.NeuralNetworkHyperModel(input_dim=X_train.shape[1], task_type='classification')

        tuner = RandomSearch(
            hypermodel,
            objective='val_auc', 
            max_trials=5,       # Reduced number of different models to try
            executions_per_trial=1, 
            directory=tuner_dir,
            project_name='nn_classification_tuning_eased', # Changed project name to avoid conflicts if old runs exist
            overwrite=True 
        )

        logger.info("Running Keras Tuner search...")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True # Reduced patience
        )

        tuner.search(
            X_train, y_train,
            epochs=20, # Reduced max epochs for each trial
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0 
        )

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0].values
        
        mapped_params = {
            'hidden_layers': best_hp['num_hidden_layers'],
            'neurons': best_hp['neurons'],
            'dropout_rate': best_hp['dropout_rate'],
            'learning_rate': best_hp['learning_rate']
        }

        self.best_params_cache[model_name] = mapped_params
        self._save_all_best_params()
        
        logger.info(f"Best hyperparameters for Neural Network Classification: {mapped_params}")
        
        return mapped_params
