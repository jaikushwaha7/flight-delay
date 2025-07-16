
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders as ce
from typing import Tuple, Set, Dict, List, Union

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DataPreprocessor:
    """
    Handles various data preprocessing steps for the flight delay dataset.
    """
    def __init__(self):
        self.label_encoders = {} # To store LabelEncoders for inverse transform if needed
        self.one_hot_encoder = None # To store the OneHotEncoder transformer
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.woe_encoder = None # To store the WOE encoder
        logger.info("DataPreprocessor initialized.")

    def handle_missing_values(self, df: pd.DataFrame, delay_cols: List[str]) -> pd.DataFrame:
        """
        Handles missing values, specifically for delay reason columns.
        Fills NaN in delay reason columns with 0.
        Drops rows where 'ARR_DELAY' is NaN (as it's our target).
        """
        logger.info("Handling missing values...")
        
        # Fill NaN in delay reason columns with 0 (assuming NaN means no delay reason from that category)
        for col in delay_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                logger.debug(f"Filled NaN in '{col}' with 0.")

        # Drop rows where 'ARR_DELAY' is NaN, as it's the target variable
        initial_rows = df.shape[0]
        if 'ARR_DELAY' in df.columns:
            df.dropna(subset=['ARR_DELAY'], inplace=True)
            rows_dropped = initial_rows - df.shape[0]
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows due to NaN values in 'ARR_DELAY'.")
        else:
            logger.warning("'ARR_DELAY' column not found, skipping NaN drop for target.")

        logger.info(f"DataFrame shape after handling missing values: {df.shape}")
        return df

    def remove_outliers_iqr(self, df: pd.DataFrame, column: str = 'ARR_DELAY', iqr_multiplier: float = 3.0) -> pd.DataFrame:
        """
        Removes outliers from a specified numerical column using the IQR method.
        """
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            logger.warning(f"Column '{column}' not found or not numeric. Skipping outlier removal.")
            return df

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        initial_rows = df.shape[0]
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
        rows_removed = initial_rows - df_filtered.shape[0]

        logger.info(f"Removed {rows_removed} outliers from '{column}' using IQR (multiplier={iqr_multiplier}).")
        logger.info(f"DataFrame shape after outlier removal: {df_filtered.shape}")
        return df_filtered

    def create_elapsed_time_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the difference between actual elapsed time and scheduled elapsed time.
        Handles missing values in time columns by dropping rows or imputing if necessary.
        """
        logger.info("Creating 'ELAPSED_TIME_DIFF' feature...")
        
        # Define time columns that must be present and numeric
        required_time_cols = ['ACTUAL_ELAPSED_TIME', 'CRS_ELAPSED_TIME']
        
        for col in required_time_cols:
            if col not in df.columns:
                logger.warning(f"Required time column '{col}' not found. Cannot create 'ELAPSED_TIME_DIFF'.")
                return df
            # Ensure they are numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential time components are NaN
        initial_rows = df.shape[0]
        df.dropna(subset=required_time_cols, inplace=True)
        rows_dropped = initial_rows - df.shape[0]
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows due to missing values in required time columns for 'ELAPSED_TIME_DIFF'.")

        if not df.empty:
            df['ELAPSED_TIME_DIFF'] = df['ACTUAL_ELAPSED_TIME'] - df['CRS_ELAPSED_TIME']
            logger.info("Created 'ELAPSED_TIME_DIFF' column.")
        else:
            logger.warning("DataFrame is empty after dropping rows for time features; 'ELAPSED_TIME_DIFF' not created.")

        return df

    def apply_cyclical_encoding(self, df: pd.DataFrame, time_columns: List[str]) -> pd.DataFrame:
        """
        Applies cyclical (sine and cosine) encoding to time-based features.
        Assumes time columns are in minutes (0-1439 for a day).
        """
        logger.info("Applying cyclical encoding to time columns...")
        for col in time_columns:
            if col in df.columns:
                # Convert to numeric, errors will be NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaNs with mean before encoding to avoid NaNs in sin/cos
                if df[col].isnull().any():
                    col_mean = df[col].mean()
                    df[col].fillna(col_mean, inplace=True)
                    logger.warning(f"Filled NaN in '{col}' with mean {col_mean:.2f} before cyclical encoding.")

                max_val = 2359 if 'TIME' in col.upper() else 1439 # Max minutes in 24 hours (HHMM format)
                if 'TIME' in col.upper(): # Heuristic for HHMM format
                    # Convert HHMM to minutes
                    df[f'{col}_MINUTES'] = (df[col] // 100) * 60 + (df[col] % 100)
                    col_to_encode = f'{col}_MINUTES'
                    max_val_for_sin_cos = 1440 # 24 * 60 minutes
                else: # Assume already in minutes or similar scale (e.g. wheels_on/off might be unix timestamp or similar, adjust max_val if needed)
                    col_to_encode = col
                    max_val_for_sin_cos = df[col].max() if df[col].max() > 0 else 1 # Avoid division by zero

                if max_val_for_sin_cos > 0:
                    df[f'{col_to_encode}_SIN'] = np.sin(2 * np.pi * df[col_to_encode] / max_val_for_sin_cos)
                    df[f'{col_to_encode}_COS'] = np.cos(2 * np.pi * df[col_to_encode] / max_val_for_sin_cos)
                    logger.debug(f"Applied cyclical encoding to '{col_to_encode}'.")
                    # Optionally drop the original column to avoid multicollinearity
                    # df.drop(columns=[col_to_encode], inplace=True, errors='ignore')
                else:
                    logger.warning(f"Max value for '{col_to_encode}' is zero, skipping cyclical encoding.")
            else:
                logger.warning(f"Time column '{col}' not found. Skipping cyclical encoding.")
        return df

    def split_city_state(self, df: pd.DataFrame, city_state_columns: List[str]) -> pd.DataFrame:
        """
        Splits 'CITY, STATE' columns into 'CITY' and 'STATE' columns.
        """
        logger.info("Splitting city and state columns...")
        for col in city_state_columns:
            if col in df.columns and df[col].astype(str).str.contains(',').any():
                df[[f'{col}_CITY', f'{col}_STATE']] = df[col].astype(str).str.split(', ', expand=True)
                df.drop(columns=[col], inplace=True)
                logger.debug(f"Split '{col}' into '{col}_CITY' and '{col}_STATE'.")
            else:
                logger.warning(f"Column '{col}' not found or does not contain 'CITY, STATE' format. Skipping split.")
        return df

    def add_weekday_weekend_columns(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Adds 'DAY_OF_WEEK' and 'IS_WEEKEND' columns from date columns.
        """
        logger.info("Adding weekday and weekend columns...")
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df['DAY_OF_WEEK'] = df[col].dt.dayofweek # Monday=0, Sunday=6
                    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int) # Saturday=5, Sunday=6
                    logger.debug(f"Added 'DAY_OF_WEEK' and 'IS_WEEKEND' from '{col}'.")
                except Exception as e:
                    logger.error(f"Error converting '{col}' to datetime or extracting day features: {e}. Skipping.", exc_info=True)
            else:
                logger.warning(f"Date column '{col}' not found. Skipping weekday/weekend features.")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies WOE (Weight of Evidence) encoding to all categorical features based on target variable FLIGHT_STATUS.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with WOE encoded categorical features.
        """
        df_copy = df.copy()
        logger.info("Applying WOE encoding to all categorical features based on FLIGHT_STATUS.")

        # Check if target variable exists
        target_variable = 'FLIGHT_STATUS'
        if target_variable not in df_copy.columns:
            logger.error(f"Target variable '{target_variable}' not found in DataFrame. Cannot apply WOE encoding.")
            return df_copy

        # Identify all categorical columns (excluding the target variable)
        categorical_columns = []
        
        # Get columns with object dtype (string/categorical)
        object_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        # Get columns with category dtype
        category_cols = df_copy.select_dtypes(include=['category']).columns.tolist()
        
        # Combine and remove target variable
        categorical_columns = list(set(object_cols + category_cols))
        if target_variable in categorical_columns:
            categorical_columns.remove(target_variable)
        
        # You can also explicitly specify columns you want to encode
        # Uncomment and modify this list if you want to be more specific:
        # categorical_columns = [
        #     'FL_DATE_day_name', 'AIRLINE', 'FL_NUMBER', 'ORIGIN', 'ORIGIN_CITY', 
        #     'DEST', 'DEST_CITY', 'origin_state', 'dest_state'
        # ]
        # categorical_columns = [col for col in categorical_columns if col in df_copy.columns]

        if not categorical_columns:
            logger.warning("No categorical columns found for WOE encoding.")
            return df_copy

        logger.info(f"Found {len(categorical_columns)} categorical columns for WOE encoding: {categorical_columns}")

        # Apply WOE encoding to all categorical columns
        try:
            # Handle missing values in target variable
            if df_copy[target_variable].isnull().any():
                logger.warning(f"Target variable '{target_variable}' contains NaN values. This may affect WOE encoding.")
                # You might want to handle NaNs in target variable before encoding
                # df_copy = df_copy.dropna(subset=[target_variable])  # Option 1: Drop rows with NaN target
                # df_copy[target_variable] = df_copy[target_variable].fillna('Unknown')  # Option 2: Fill NaNs
            
            # Handle missing values in categorical columns
            # for col in categorical_columns:
            #     if df_copy[col].isnull().any():
            #         logger.info(f"Column '{col}' contains NaN values. Filling with 'Missing' before WOE encoding.")
            #         df_copy[col] = df_copy[col].fillna('Missing')
            
            # Initialize WOE encoder
            self.woe_encoder = ce.WOEEncoder(cols=categorical_columns, handle_unknown='value', handle_missing='value')
            
            # Apply WOE encoding
            df_encoded = self.woe_encoder.fit_transform(df_copy[categorical_columns], df_copy[target_variable])
            
            # Replace original categorical columns with WOE encoded versions
            # Add '_WOE' suffix to distinguish encoded columns
            woe_columns = {}
            for i, col in enumerate(categorical_columns):
                woe_col_name = f'{col}_WOE'
                df_copy[woe_col_name] = df_encoded.iloc[:, i]
                woe_columns[col] = woe_col_name
            
            logger.info(f"Successfully applied WOE encoding to {len(categorical_columns)} categorical columns.")
            logger.info(f"Created WOE encoded columns: {list(woe_columns.values())}")
            
            # Drop original categorical columns
            df_copy = df_copy.drop(columns=categorical_columns)
            logger.info(f"Dropped original categorical columns: {categorical_columns}")
            
        except Exception as e:
            logger.error(f"Error applying WOE encoding: {e}", exc_info=True)
            return df_copy
        
        return df_copy

    def identify_high_correlation_pairs(self, df: pd.DataFrame, threshold: float = 0.95) -> Tuple[Set[str], Dict[Tuple[str, str], float]]:
        """
        Identifies pairs of highly correlated features and suggests one to drop.
        Returns a set of columns to drop and a dictionary of correlated pairs.
        """
        logger.info(f"Identifying high correlation features (threshold={threshold})...")
        
        # Select only numeric columns for correlation calculation
        numeric_df = df.select_dtypes(include=np.number)
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation analysis.")
            return set(), {}

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        correlated_pairs = {}

        for i in range(len(upper_tri.columns)):
            for j in range(i + 1, len(upper_tri.columns)):
                col1 = upper_tri.columns[i]
                col2 = upper_tri.columns[j]
                correlation_value = upper_tri.iloc[i, j]

                if pd.notna(correlation_value) and correlation_value >= threshold:
                    correlated_pairs[(col1, col2)] = correlation_value
                    # Simple heuristic: remove the second column in the pair
                    to_drop.add(col2)
                    logger.debug(f"High correlation found: {col1} and {col2} (Corr: {correlation_value:.2f}). Suggesting to drop {col2}.")

        logger.info(f"Found {len(to_drop)} columns to drop due to high correlation.")
        return to_drop, correlated_pairs

    def exclude_columns(self, df: pd.DataFrame, columns_to_exclude: List[str]) -> pd.DataFrame:
        """
        Excludes specified columns from the DataFrame.
        """
        logger.info("Excluding specified columns...")
        existing_cols_to_drop = [col for col in columns_to_exclude if col in df.columns]
        if existing_cols_to_drop:
            df.drop(columns=existing_cols_to_drop, inplace=True)
            logger.info(f"Excluded columns: {existing_cols_to_drop}")
        else:
            logger.info("No specified columns found to exclude.")
        return df

    def create_classification_target(self, df: pd.DataFrame, delay_column: str, delay_threshold_minutes: int) -> pd.DataFrame:
        """
        Creates a binary classification target based on arrival delay.
        1 if ARR_DELAY > delay_threshold_minutes, 0 otherwise.
        """
        if delay_column not in df.columns:
            logger.error(f"Delay column '{delay_column}' not found for creating classification target.")
            return df
        
        logger.info(f"Creating classification target: FLIGHT_STATUS_CLASSIFICATION (1 if {delay_column} > {delay_threshold_minutes} mins, else 0).")
        df['FLIGHT_STATUS_CLASSIFICATION'] = (df[delay_column] > delay_threshold_minutes).astype(int)
        logger.info(f"Classification target value counts:\n{df['FLIGHT_STATUS_CLASSIFICATION'].value_counts().to_string()}")
        return df
