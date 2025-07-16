import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tensorflow as tf
import wandb # NEW: Import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class ModelExplainer:
    """
    Handles model explainability for trained machine learning models.
    """
    def __init__(self, output_dir="plots/explainability", wandb_run=None): # NEW: Add wandb_run
        self.output_dir = output_dir
        self.wandb_run = wandb_run # NEW: Store wandb_run
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ModelExplainer initialized. Explainability plots will be saved to '{self.output_dir}'.")

    def plot_feature_importance(self, model, feature_names: list, filename: str):
        """
        Plots feature importance for tree-based models (RandomForest).

        Args:
            model: Trained scikit-learn model (must have .feature_importances_ attribute).
            feature_names (list): List of feature names corresponding to model's input.
            filename (str): Name of the file to save the plot.
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model of type {type(model).__name__} does not have 'feature_importances_'. Skipping feature importance plot.")
            return

        try:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20)) # Top 20 features
            plt.title(f'Top 20 Feature Importance for {type(model).__name__}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved feature importance plot to {plot_path}")

            # NEW: Log plot to WandB
            if self.wandb_run:
                self.wandb_run.log({f"Explainability/{filename.replace('.png', '')}": wandb.Image(plot_path)})
                logger.info(f"Feature importance plot logged to WandB for {type(model).__name__}.")

        except Exception as e:
            logger.error(f"Error generating feature importance plot: {e}")

    def plot_coefficient_importance(self, model, feature_names: list, filename: str):
        """
        Plots coefficients for linear models (LogisticRegression, Ridge).

        Args:
            model: Trained scikit-learn linear model (must have .coef_ attribute).
            feature_names (list): List of feature names.
            filename (str): Name of the file to save the plot.
        """
        if not hasattr(model, 'coef_'):
            logger.warning(f"Model of type {type(model).__name__} does not have 'coef_'. Skipping coefficient importance plot.")
            return

        try:
            coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_

            coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
            coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
            coef_df = coef_df.sort_values(by='abs_coefficient', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='coefficient', y='feature', data=coef_df.head(20)) # Top 20 features by absolute coefficient
            plt.title(f'Top 20 Feature Coefficients for {type(model).__name__}')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved coefficient importance plot to {plot_path}")

            # NEW: Log plot to WandB
            if self.wandb_run:
                self.wandb_run.log({f"Explainability/{filename.replace('.png', '')}": wandb.Image(plot_path)})
                logger.info(f"Coefficient importance plot logged to WandB for {type(model).__name__}.")

        except Exception as e:
            logger.error(f"Error generating coefficient importance plot: {e}")


    def plot_shap_summary(self, model, X: pd.DataFrame, filename: str, task_type: str = 'regression', num_features: int = 20):
        """
        Generates and saves a SHAP summary plot.

        Args:
            model: The trained model (scikit-learn or Keras).
            X (pd.DataFrame): The data used to generate SHAP values (e.g., X_test or a sample).
            filename (str): Name of the file to save the plot.
            task_type (str): 'regression' or 'classification'.
            num_features (int): Number of features to display in the summary plot.
        """
        try:
            if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, (LogisticRegression, Ridge)):
                explainer = shap.LinearExplainer(model, X)
            elif isinstance(model, tf.keras.Model):
                explainer = shap.DeepExplainer(model, X.values)
            else:
                logger.warning(f"SHAP Explainer for model type {type(model).__name__} not explicitly supported/optimized. Using KernelExplainer (can be slow).")
                predict_fn = model.predict_proba if task_type == 'classification' and hasattr(model, 'predict_proba') else model.predict
                explainer = shap.KernelExplainer(predict_fn, shap.sample(X, 100, random_state=42))


            if isinstance(explainer, shap.DeepExplainer) and task_type == 'classification':
                shap_values = explainer.shap_values(X.values)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            elif isinstance(explainer, shap.LinearExplainer) and task_type == 'classification':
                shap_values = explainer.shap_values(X.values)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            elif isinstance(explainer, shap.KernelExplainer) and task_type == 'classification':
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X.values if isinstance(explainer, (shap.DeepExplainer, shap.LinearExplainer)) else X)

            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=num_features)
            plt.title(f'SHAP Summary Bar Plot for {type(model).__name__} ({task_type.capitalize()})')
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved SHAP summary bar plot to {plot_path}")

            # NEW: Log plot to WandB
            if self.wandb_run:
                self.wandb_run.log({f"Explainability/{filename.replace('.png', '')}_bar": wandb.Image(plot_path)})
                logger.info(f"SHAP summary bar plot logged to WandB for {type(model).__name__}.")


            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X, show=False, max_display=num_features)
            plt.title(f'SHAP Summary Dot Plot for {type(model).__name__} ({task_type.capitalize()})')
            plt.tight_layout()
            dot_filename = filename.replace(".png", "_dot.png")
            dot_plot_path = os.path.join(self.output_dir, dot_filename)
            plt.savefig(dot_plot_path)
            plt.close()
            logger.info(f"Saved SHAP summary dot plot to {dot_plot_path}")

            # NEW: Log plot to WandB
            if self.wandb_run:
                self.wandb_run.log({f"Explainability/{dot_filename.replace('.png', '')}_dot": wandb.Image(dot_plot_path)})
                logger.info(f"SHAP summary dot plot logged to WandB for {type(model).__name__}.")

        except Exception as e:
            logger.error(f"Error generating SHAP summary plot for {type(model).__name__}: {e}", exc_info=True)


    def plot_shap_dependence(self, model, X: pd.DataFrame, feature: str, filename: str, task_type: str = 'regression', interaction_feature: str = None):
        """
        Generates and saves a SHAP dependence plot for a specific feature.

        Args:
            model: The trained model (scikit-learn or Keras).
            X (pd.DataFrame): The data used to generate SHAP values (e.g., X_test or a sample).
            feature (str): The name of the feature for which to plot dependence.
            filename (str): Name of the file to save the plot.
            task_type (str): 'regression' or 'classification'.
            interaction_feature (str, optional): A feature to color the dependence plot by, revealing interactions.
        """
        if feature not in X.columns:
            logger.warning(f"Feature '{feature}' not found in X. Skipping SHAP dependence plot.")
            return
        if interaction_feature and interaction_feature not in X.columns:
            logger.warning(f"Interaction feature '{interaction_feature}' not found in X. Skipping SHAP dependence plot with interaction.")
            interaction_feature = None

        try:
            if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, (LogisticRegression, Ridge)):
                explainer = shap.LinearExplainer(model, X)
            elif isinstance(model, tf.keras.Model):
                explainer = shap.DeepExplainer(model, X.values)
            else:
                predict_fn = model.predict_proba if task_type == 'classification' and hasattr(model, 'predict_proba') else model.predict
                explainer = shap.KernelExplainer(predict_fn, shap.sample(X, 100, random_state=42))

            if isinstance(explainer, shap.DeepExplainer) and task_type == 'classification':
                shap_values = explainer.shap_values(X.values)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            elif isinstance(explainer, shap.LinearExplainer) and task_type == 'classification':
                shap_values = explainer.shap_values(X.values)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            elif isinstance(explainer, shap.KernelExplainer) and task_type == 'classification':
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X.values if isinstance(explainer, (shap.DeepExplainer, shap.LinearExplainer)) else X)

            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)


            plt.figure(figsize=(10, 7))
            shap.dependence_plot(
                ind=feature,
                shap_values=shap_values,
                features=X,
                feature_names=X.columns.tolist(),
                interaction_index=interaction_feature,
                show=False,
                title=f'SHAP Dependence Plot for {feature} ({type(model).__name__})'
            )
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved SHAP dependence plot for '{feature}' to {plot_path}")

            # NEW: Log plot to WandB
            if self.wandb_run:
                self.wandb_run.log({f"Explainability/{filename.replace('.png', '')}": wandb.Image(plot_path)})
                logger.info(f"SHAP dependence plot logged to WandB for '{feature}' ({type(model).__name__}).")

        except Exception as e:
            logger.error(f"Error generating SHAP dependence plot for feature '{feature}': {e}", exc_info=True)