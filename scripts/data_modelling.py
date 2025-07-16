
import pandas as pd
import numpy as np
import logging

# For Data Splitting and Preprocessing
from sklearn.model_selection import KFold

# For Regression Models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# For Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier # NEW: Import for baseline classification model

# For Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError, BinaryAccuracy, AUC # Ensure AUC is imported
from tensorflow.keras.callbacks import EarlyStopping # Explicitly import EarlyStopping

# nn_utils for using in tuning and modelling
from scripts.nn_utils import build_nn_model # This is where your build_nn_model should be located

# Import ModelTuner
from scripts.model_tuning import ModelTuner


# Configure a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DataModeling:
    """
    Handles training of various regression and classification models.
    """
    def __init__(self, tuner: ModelTuner): # Accept a ModelTuner instance
        self.tuner = tuner
        logger.info("DataModeling initialized.")

    def run_baseline_model(self, y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, None]:
        """
        Runs a baseline model that predicts the average of the training target (for regression).

        Args:
            y_train (pd.Series): Training target values.
            y_test (pd.Series): Test target values.

        Returns:
            tuple[np.ndarray, None]: Predicted values (no model object for baseline).
        """
        logger.info("--- Running Baseline Regression Model (Average Prediction) ---")
        y_pred_baseline = np.full(len(y_test), y_train.mean())
        return y_pred_baseline, None # No model object to return for baseline (as it's a simple constant)

    def run_baseline_model_classification(self, y_train_val: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, DummyClassifier]:
        """
        Trains and returns a baseline classification model (predicts majority class).

        Args:
            y_train_val (pd.Series): Training + Validation target values to find majority class.
            y_test (pd.Series): Test target values (used for determining prediction array size).

        Returns:
            tuple[np.ndarray, DummyClassifier]: Predicted class labels for the test set, and the trained DummyClassifier model.
        """
        logger.info("--- Running Baseline Classification Model (Majority Class Classifier) ---")

        # Initialize DummyClassifier with 'most_frequent' strategy
        baseline_model = DummyClassifier(strategy='most_frequent', random_state=42)

        # Fit the model. X is not truly used, but required for the fit method.
        # We just need to fit on y_train_val to determine the most frequent class.
        # We create a dummy X with the same number of samples as y_train_val.
        dummy_X = np.zeros((len(y_train_val), 1))
        baseline_model.fit(dummy_X, y_train_val)

        # Predict on a dummy X for the test set to get the predictions
        dummy_X_test = np.zeros((len(y_test), 1))
        y_pred_baseline = baseline_model.predict(dummy_X_test)

        majority_class = baseline_model.predict(np.array([[0]])) 
        if majority_class[0] == 0:
            y_pred_proba_baseline = np.zeros(len(y_test))
        else: # majority_class[0] == 1
            y_pred_proba_baseline = np.ones(len(y_test))

        logger.info(f"Baseline (majority class) for training data: {majority_class[0]}")
        logger.info("Baseline Classification Model training complete.")

        return y_pred_baseline, baseline_model # Return actual predictions and the model itself

    def run_ridge_regression(self, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> Ridge:
        """
        Trains a Ridge Regression model using hyperparameter tuning on the combined
        training and validation set.

        Args:
            X_train_val (pd.DataFrame): Training + Validation features.
            y_train_val (pd.Series): Training + Validation target.

        Returns:
            Ridge: The best trained Ridge model.
        """
        logger.info("\n--- Running Ridge Regression ---")
        ridge = Ridge(random_state=42)
        # Eased out param_grid for faster tuning
        param_grid = {'alpha': [1, 10]} # Smaller, more focused grid

        # Attempt to load best params first
        best_params_loaded = self.tuner.get_best_params("ridge_regression")

        if best_params_loaded:
            logger.info("Using loaded best parameters for Ridge Regression to train final model.")
            best_ridge = Ridge(**best_params_loaded, random_state=42)
            best_ridge.fit(X_train_val, y_train_val)
        else:
            logger.info("No saved best parameters found. Performing full hyperparameter tuning for Ridge Regression.")
            best_ridge, _, _ = self.tuner.tune_model(
                model=ridge,
                param_grid=param_grid,
                X=X_train_val, # Use X_train_val for tuning
                y=y_train_val, # Use y_train_val for tuning
                model_name="ridge_regression",
                scoring='neg_mean_absolute_error' # Ensure correct scoring for regression
            )
        return best_ridge

    def run_random_forest_regression(self, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> RandomForestRegressor:
        """
        Trains a Random Forest Regressor using hyperparameter tuning on the combined
        training and validation set.

        Args:
            X_train_val (pd.DataFrame): Training + Validation features.
            y_train_val (pd.Series): Training + Validation target.

        Returns:
            RandomForestRegressor: The best trained Random Forest model.
        """
        logger.info("\n--- Running Random Forest Regression ---")
        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        # Eased out param_grid for faster tuning
        param_grid = {
            'n_estimators': [50, 100], # Fewer estimators
            'max_depth': [5, 10],      # Fewer max depths
            'min_samples_split': [2]   # Single value or very few
        }

        # Attempt to load best params first
        best_params_loaded = self.tuner.get_best_params("random_forest_regression")

        if best_params_loaded:
            logger.info("Using loaded best parameters for Random Forest Regression to train final model.")
            best_rf = RandomForestRegressor(**best_params_loaded, random_state=42, n_jobs=-1)
            best_rf.fit(X_train_val, y_train_val)
        else:
            logger.info("No saved best parameters found. Performing full hyperparameter tuning for Random Forest Regression.")
            best_rf, _, _ = self.tuner.tune_model(
                model=rf_model,
                param_grid=param_grid,
                X=X_train_val, # Use X_train_val for tuning
                y=y_train_val, # Use y_train_val for tuning
                model_name="random_forest_regression",
                search_method='random', # Using random search
                n_iter=3, # Reduced number of iterations for random search
                scoring='neg_mean_absolute_error' # Ensure correct scoring for regression
            )
        return best_rf

    def run_logistic_regression(self, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> LogisticRegression:
        """
        Trains a Logistic Regression model using hyperparameter tuning for classification.

        Args:
            X_train_val (pd.DataFrame): Training + Validation features.
            y_train_val (pd.Series): Training + Validation target.

        Returns:
            LogisticRegression: The best trained Logistic Regression model.
        """
        logger.info("\n--- Running Logistic Regression (Classification) ---")
        log_reg = LogisticRegression(random_state=42, solver='liblinear', max_iter=500, class_weight='balanced')

        # Eased out param_grid for faster tuning
        param_grid = {
            'C': [1, 10],       # Fewer C values
            'penalty': ['l1']   # Single penalty type
        }

        best_params_loaded = self.tuner.get_best_params("logistic_regression")

        if best_params_loaded:
            logger.info("Using loaded best parameters for Logistic Regression to train final model.")
            best_log_reg = LogisticRegression(**best_params_loaded, random_state=42, solver='liblinear', max_iter=500)
            best_log_reg.fit(X_train_val, y_train_val)
        else:
            logger.info("No saved best parameters found. Performing full hyperparameter tuning for Logistic Regression.")
            best_log_reg, _, _ = self.tuner.tune_model(
                model=log_reg,
                param_grid=param_grid,
                X=X_train_val,
                y=y_train_val,
                model_name="logistic_regression",
                scoring='f1' # Common scoring for classification, especially with imbalance
            )
        return best_log_reg

    def run_neural_network(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series, task_type: str = 'regression') -> tf.keras.models.Sequential:
        """
        Trains a Neural Network model for either regression or classification.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation target.
            task_type (str): 'regression' or 'classification'.

        Returns:
            tf.keras.models.Sequential: The trained Keras model.
        """
        logger.info(f"\n--- Running Neural Network ({task_type.capitalize()}) ---")
        tf.random.set_seed(42)
        np.random.seed(42)

        # Default params, will be overridden by best_hyperparameters if tuning occurs
        nn_model_params = {
            'input_dim': X_train.shape[1],
            'hidden_layers': 2,
            'neurons': 64,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'task_type': task_type # Pass task type to build_nn_model
        }

        # Check if we have best params for 'neural_network_regression' or 'neural_network_classification'
        model_name_for_params = f"neural_network_{task_type}"
        best_params_loaded = self.tuner.get_best_params(model_name_for_params)

        if best_params_loaded:
            logger.info(f"Using loaded best parameters for {model_name_for_params} to train final model.")
            # Merge loaded params with fixed ones (like input_dim, task_type)
            nn_model_params.update(best_params_loaded)
            # Ensure task_type is correctly set for build_nn_model
            nn_model_params['task_type'] = task_type 

        nn_model = build_nn_model(**nn_model_params)

        logger.info(f"Neural Network Model Summary ({task_type}):\n{nn_model.summary()}")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5, # Reduced patience for faster convergence
            restore_best_weights=True
        )

        history = nn_model.fit(X_train, y_train,
                               validation_data=(X_val, y_val),
                               epochs=50, # Reduced max epochs with early stopping
                               batch_size=32,
                               verbose=0, # Set to 1 for progress bar during training
                               callbacks=[early_stopping])

        logger.info(f"Neural Network {task_type.capitalize()} Model training complete.")
        return nn_model

    def run_neural_network_classification(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> tf.keras.models.Sequential:
        """Trains and returns a Neural Network Classification model."""
        logger.info("Training Neural Network Classification Model...")
        input_dim = X_train.shape[1]

        best_hyperparameters = None
        if self.tuner:
            best_hyperparameters = self.tuner.tune_neural_network_classification(X_train, y_train, X_val, y_val)
            logger.info(f"Neural Network best hyperparameters: {best_hyperparameters}")

        if best_hyperparameters:
            # Use the build_nn_model function with the tuned hyperparameters
            model = build_nn_model(
                input_dim=input_dim,
                hidden_layers=best_hyperparameters['hidden_layers'],
                neurons=best_hyperparameters['neurons'],
                dropout_rate=best_hyperparameters['dropout_rate'],
                learning_rate=best_hyperparameters['learning_rate'],
                task_type='classification' # Explicitly set task_type
            )
        else:
            # Default model for non-tuned scenario (if tuner is None or returns no params)
            # These defaults should align with reasonable starting points for build_nn_model
            model = build_nn_model(
                input_dim=input_dim,
                hidden_layers=2,
                neurons=64,
                dropout_rate=0.3,
                learning_rate=0.001,
                task_type='classification'
            )
        
        # Define epochs and early stopping based on whether tuning occurred
        # Note: epochs might be tuned in Keras Tuner, but here we just use early stopping to control
        epochs = 50 # Start with a moderate number, EarlyStopping will cut it short
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Reduced patience

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0 # Set to 1 for progress bar during training
        )
        logger.info("Neural Network Classification Model training complete.")
        return model
