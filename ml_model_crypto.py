# Data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
# %matplotlib inline

# Machine learning model from scikit-learn
from sklearn.linear_model import LinearRegression

# To ignore warnings (optional)
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pandas_ta as ta  # Make sure you have installed pandas_ta (pip install pandas_ta)

# Import necessary libraries and metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Replace 'crypto_data.csv' with the actual filename
df = pd.read_csv('crypto_data.csv')
# df.head()

def preprocess_crypto_df(df):
    """
    Preprocess the cryptocurrency DataFrame.

    Expected DataFrame columns:
    - 'Time': String in the format "dd-mm-yyyy HH:MM" (e.g., "01-01-2018 00:00")
    - 'Open', 'High', 'Low', 'Close', 'Volume': Numeric values as strings or numbers.

    The function will:
    - Convert the 'Time' column to a datetime object.
    - Drop rows with invalid dates.
    - Set 'Time' as the DataFrame index.
    - Ensure the numeric columns are in a proper numeric format.
    - Drop rows with missing values in numeric columns.
    - Sort the DataFrame by time.

    Returns:
        A cleaned and preprocessed DataFrame.
    """

    # Make a copy of the DataFrame to avoid modifying the original data
    df = df.copy()

    # Convert 'Time' column to datetime using the expected format
    # errors='coerce' will convert any invalid parsing into NaT (Not a Time)
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M', errors='coerce')

    # Drop any rows where 'Time' conversion failed (i.e., contains NaT)
    df.dropna(subset=['Time'], inplace=True)

    # Set the 'Time' column as the index of the DataFrame
    df.set_index('Time', inplace=True)

    # List of columns expected to be numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Convert each numeric column to a proper numeric type
    # errors='coerce' will set any non-convertible values to NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that have any missing values in the numeric columns
    df.dropna(subset=numeric_cols, inplace=True)

    # Sort the DataFrame by the time index to ensure chronological order
    df.sort_index(inplace=True)

    # Return the cleaned and preprocessed DataFrame
    return df

def add_technical_features(df):
    """
    Adds technical indicators and features from the pandas_ta library to the DataFrame.

    The function assumes the DataFrame contains the following columns:
      - 'Open'
      - 'High'
      - 'Low'
      - 'Close'
      - 'Volume'

    It will add:
      - Multiple EMAs for periods: 5, 20, 45, 50, 100, 200.
      - RSI (Relative Strength Index) with a default period of 14.
      - MACD (Moving Average Convergence Divergence) along with its histogram and signal.
      - Bollinger Bands (Lower, Middle, and Upper bands).
      - Stochastic Oscillator.
      - ATR (Average True Range) with a period of 14.
      - 'Next_Close': The closing price of the next candle (target for prediction).

    Finally, it drops any rows with missing values (e.g., at the beginning or end due to shifting/rolling calculations).

    Parameters:
      df (pd.DataFrame): Input DataFrame containing crypto price data.

    Returns:
      pd.DataFrame: DataFrame enriched with new technical features.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    df = df.copy()

    # ------------------------------
    # Add Multiple Exponential Moving Averages (EMAs)
    # ------------------------------
    ema_periods = [5, 20, 45, 50, 100, 200]
    for period in ema_periods:
        # Calculate EMA for the given period and add as a new column, e.g., 'EMA_5'
        df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)

    # ------------------------------
    # Add Relative Strength Index (RSI)
    # ------------------------------
    # Default period for RSI is 14
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # ------------------------------
    # Add MACD (Moving Average Convergence Divergence)
    # ------------------------------
    # MACD returns a DataFrame with columns: MACD, MACDh (histogram), and MACDs (signal)
    macd_df = ta.macd(df['Close'])
    # Concatenate the MACD columns with the main DataFrame
    df = pd.concat([df, macd_df], axis=1)

    # ------------------------------
    # Add Bollinger Bands
    # ------------------------------
    # Bollinger Bands returns a DataFrame with lower band (BBL), middle band (BBM), upper band (BBU), etc.
    bbands_df = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands_df], axis=1)

    # ------------------------------
    # Add Stochastic Oscillator
    # ------------------------------
    # Using high, low, and close prices. Returns %K and %D values.
    stoch_df = ta.stoch(df['High'], df['Low'], df['Close'])
    df = pd.concat([df, stoch_df], axis=1)

    # ------------------------------
    # Add Average True Range (ATR)
    # ------------------------------
    # ATR is computed over a period of 14 by default.
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # ------------------------------
    # Create the Target Variable: Next Candle's Closing Price
    # ------------------------------
    # Shift the 'Close' column by -1 to get the next candle's closing price
    df['Next_Close'] = df['Close'].shift(-1)

    # Drop rows with any missing values generated by rolling calculations or shifting
    df.dropna(inplace=True)

    return df

preprocessed_df = preprocess_crypto_df(df)
featured_df = add_technical_features(preprocessed_df)
# print(featured_df.head())

# Assume you have a DataFrame 'featured_df' that was returned from your add_technical_features() function.

# Determine the index to split the data (80% training, 20% testing)
train_size = int(0.8 * len(featured_df))

# Create training and testing sets by slicing based on time order
train_data = featured_df.iloc[:train_size]
test_data = featured_df.iloc[train_size:]

# Define the target variable (Next_Close) and feature columns.
# Features: all columns except 'Next_Close'
feature_cols = featured_df.columns.difference(['Next_Close']).tolist()

# Split features (X) and target (y) for training and testing sets.
X_train = train_data[feature_cols]
y_train = train_data['Next_Close']
X_test = test_data[feature_cols]
y_test = test_data['Next_Close']

# Print the shapes to verify the split
print("Training set shape (features, target):", X_train.shape, y_train.shape)
print("Testing set shape (features, target):", X_test.shape, y_test.shape)

# -----------------------------
# Random Forest Regressor
# -----------------------------
# Initialize the Random Forest model with a fixed random state for reproducibility.
rf_model = RandomForestRegressor(random_state=42)

# Train the model on the training data.
rf_model.fit(X_train, y_train)

# Predict the target variable on the test set.
rf_predictions = rf_model.predict(X_test)

# Evaluate model performance using:
# - Mean Squared Error (MSE)
# - Mean Absolute Error (MAE)
# - R-squared (R²) score
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Print out the performance metrics for Random Forest.
print("Random Forest Performance:")
print(f"Mean Squared Error: {rf_mse:.2f}")
print(f"Mean Absolute Error: {rf_mae:.2f}")
print(f"R-squared: {rf_r2:.2f}")

# -----------------------------
# XGBoost Regressor
# -----------------------------
# Initialize the XGBoost model.
# The 'reg:squarederror' objective is used for regression.
# ----- -----------------------------------------------------------------------------------------
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model on the training data.
xgb_model.fit(X_train, y_train)

# Predict the target variable on the test set.
xgb_predictions = xgb_model.predict(X_test)

# Evaluate model performance using the same metrics as before.
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

# Print out the performance metrics for XGBoost.
print("\nXGBoost Performance:")
print(f"Mean Squared Error: {xgb_mse:.2f}")
print(f"Mean Absolute Error: {xgb_mae:.2f}")
print(f"R-squared: {xgb_r2:.2f}")

# --------------------------------------
# Helper Function: Create Sequences for LSTM
# --------------------------------------
def create_sequences(X, y, seq_length):
    """
    Create sequences of features and corresponding target values.

    Parameters:
      X (DataFrame): DataFrame containing feature data.
      y (Series or array): Target variable (e.g., next candle's close price).
      seq_length (int): Number of timesteps to include in each sequence.

    Returns:
      X_seq (numpy array): Array of shape (num_samples, seq_length, num_features)
      y_seq (numpy array): Array of target values corresponding to each sequence.
    """
    X_seq, y_seq = [], []
    # Loop over the dataset to create sequences
    for i in range(len(X) - seq_length):
        # Extract a block of consecutive rows as one sequence
        X_seq.append(X.iloc[i:i+seq_length].values)
        # The target is the value right after this sequence
        y_seq.append(y.iloc[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# --------------------------------------
# Step 1: Create Sequences for Training and Testing
# --------------------------------------
# Define the number of past timesteps to use as input (e.g., 10)
sequence_length = 10

# Build training sequences from your training set
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
# Build testing sequences from your test set
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

# --------------------------------------
# Step 2: Build the LSTM Model
# --------------------------------------
model = Sequential()

# First LSTM layer with 50 units, returning sequences to stack another LSTM layer
model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])))
model.add(Dropout(0.2))  # Dropout to reduce overfitting

# Second LSTM layer to further capture temporal patterns
model.add(LSTM(50, activation='tanh'))
model.add(Dropout(0.2))

# Final Dense layer to output the predicted next closing price
model.add(Dense(1))

# Compile the model with Mean Squared Error loss and the Adam optimizer
model.compile(optimizer='adam', loss='mse')

# Print model summary to understand the architecture
model.summary()

# --------------------------------------
# Step 3: Train the LSTM Model
# --------------------------------------
# Train the model on the training sequences. Here we use 20 epochs and a batch size of 32.
history = model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, validation_split=0.1)

# --------------------------------------
# Step 4: Evaluate the Model
# --------------------------------------
# Predict the target on the test sequences
lstm_predictions = model.predict(X_test_seq)

# Calculate evaluation metrics: Mean Squared Error, Mean Absolute Error, and R-squared
lstm_mse = mean_squared_error(y_test_seq, lstm_predictions)
lstm_mae = mean_absolute_error(y_test_seq, lstm_predictions)
lstm_r2 = r2_score(y_test_seq, lstm_predictions)

# Print the performance metrics
print("LSTM Model Performance:")
print(f"Mean Squared Error: {lstm_mse:.2f}")
print(f"Mean Absolute Error: {lstm_mae:.2f}")
print(f"R-squared: {lstm_r2:.2f}")