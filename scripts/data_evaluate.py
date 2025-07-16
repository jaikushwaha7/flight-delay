import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import pickle
import logging
import tensorflow as tf
import wandb # NEW: Import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ModelEvaluator:
    """
    Handles evaluation and logging of machine learning model performance.
    """
    def __init__(self, output_dir="models", plots_dir="plots/model_evaluation", wandb_run=None): # NEW: Add wandb_run
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        self.wandb_run = wandb_run # NEW: Store wandb_run
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_df = pd.DataFrame(columns=['Model', 'Stage', 'Metric', 'Value'])
        logger.info(f"ModelEvaluator initialized. Models will be saved to '{self.output_dir}', plots to '{self.plots_dir}'.")


    def save_model(self, model, filename_prefix: str):
        """
        Saves a trained model to the specified output directory.
        Handles both scikit-learn models (pickle) and Keras models (HDF5).

        Args:
            model: The trained model object.
            filename_prefix (str): Prefix for the filename (e.g., 'logistic_regression_model').
        """
        if isinstance(model, tf.keras.Model):
            file_path = os.path.join(self.output_dir, f"{filename_prefix}.h5")
            model.save(file_path)
            logger.info(f"Keras model saved to {file_path}")
        else:
            file_path = os.path.join(self.output_dir, f"{filename_prefix}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Scikit-learn model saved to {file_path}")

        # NEW: Log model artifact to WandB
        if self.wandb_run:
            artifact = wandb.Artifact(name=f"{filename_prefix}-model", type="model")
            artifact.add_file(file_path)
            self.wandb_run.log_artifact(artifact)
            logger.info(f"Model artifact '{filename_prefix}' logged to WandB.")


    def load_model(self, filename_prefix: str):
        """
        Loads a trained model from the specified output directory.
        Handles both scikit-learn models (pickle) and Keras models (HDF5).

        Args:
            filename_prefix (str): Prefix of the filename (e.g., 'logistic_regression_model').

        Returns:
            The loaded model object, or None if not found/error.
        """
        keras_path = os.path.join(self.output_dir, f"{filename_prefix}.h5")
        sklearn_path = os.path.join(self.output_dir, f"{filename_prefix}.pkl")

        if os.path.exists(keras_path):
            try:
                # Keras model loading requires custom objects if used
                # For this pipeline, assuming standard layers
                model = tf.keras.models.load_model(keras_path)
                logger.info(f"Keras model loaded from {keras_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading Keras model from {keras_path}: {e}")
                return None
        elif os.path.exists(sklearn_path):
            try:
                with open(sklearn_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Scikit-learn model loaded from {sklearn_path}")
                return model
            except Exception as e:
                logger.error(f"Error loading scikit-learn model from {sklearn_path}: {e}")
                return None
        else:
            logger.info(f"No model found for '{filename_prefix}' at {self.output_dir}")
            return None

    def evaluate_classification_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, stage: str = "Test"):
        """
        Evaluates a classification model and logs various metrics.

        Args:
            model_name (str): Name of the model being evaluated.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
            stage (str): The stage of evaluation (e.g., "Train", "Validation", "Test").
        """
        logger.info(f"--- {model_name} Evaluation ({stage} Set) ---")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = 0.0
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC: {e}. Ensure y_true contains at least two classes.")

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        }

        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            # NEW: Log metrics to WandB
            if self.wandb_run:
                self.wandb_run.log({f"{model_name}/{stage}_{metric_name.lower().replace(' ', '_')}": value})

            # Append to internal results_df
            self.results_df.loc[len(self.results_df)] = [model_name, stage, metric_name, value]

        logger.info("\nClassification Report:\n" + classification_report(y_true, y_pred, zero_division=0))

    def plot_confusion_matrix(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, stage: str = "Test"):
        """
        Plots and saves the confusion matrix.

        Args:
            model_name (str): Name of the model.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            stage (str): The stage of evaluation.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix for {model_name} ({stage} Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix_{stage.lower()}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {plot_path}")

        # NEW: Log plot to WandB
        if self.wandb_run:
            self.wandb_run.log({f"{model_name}/{stage}_Confusion_Matrix": wandb.Image(plot_path)})
            logger.info(f"Confusion Matrix plot logged to WandB for {model_name}.")


    def plot_roc_curve(self, model_name: str, y_true: np.ndarray, y_pred_proba: np.ndarray, stage: str = "Test"):
        """
        Plots and saves the ROC curve.

        Args:
            model_name (str): Name of the model.
            y_true (np.ndarray): True labels.
            y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
            stage (str): The stage of evaluation.
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name} ({stage} Set)')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plot_path = os.path.join(self.plots_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve_{stage.lower()}.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved ROC curve plot to {plot_path}")

            # NEW: Log plot to WandB
            if self.wandb_run:
                self.wandb_run.log({f"{model_name}/{stage}_ROC_Curve": wandb.Image(plot_path)})
                logger.info(f"ROC Curve plot logged to WandB for {model_name}.")

        except ValueError as e:
            logger.warning(f"Could not plot ROC curve for {model_name}: {e}. Ensure y_true contains at least two classes.")

    def display_results_table(self):
        """Displays the collected evaluation results in a formatted table."""
        if not self.results_df.empty:
            logger.info("\n--- Overall Model Evaluation Results ---")
            # Pivot table for better readability
            pivot_df = self.results_df.pivot_table(
                index=['Model', 'Stage'],
                columns='Metric',
                values='Value',
                aggfunc='first'
            ).round(4)
            logger.info(f"\n{pivot_df.to_string()}")
        else:
            logger.info("\nNo evaluation results to display.")
