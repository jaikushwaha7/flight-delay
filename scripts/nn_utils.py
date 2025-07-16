import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError, BinaryAccuracy, AUC # Ensure AUC is imported for classification metrics

def build_nn_model(input_dim, hidden_layers=1, neurons=64, dropout_rate=0.0, learning_rate=0.001, task_type='regression'):
    """
    Builds a Keras Sequential model for regression or classification.
    """
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))

    for _ in range(hidden_layers - 1): # Add additional hidden layers
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout_rate))

    if task_type == 'regression':
        model.add(Dense(1, activation='linear')) # Single output for regression
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=MeanSquaredError(),
                      metrics=[MeanAbsoluteError()])
    elif task_type == 'classification':
        model.add(Dense(1, activation='sigmoid')) # Single output for binary classification
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy(), AUC()]) # Add AUC for classification
    else:
        raise ValueError("task_type must be 'regression' or 'classification'")

    return model