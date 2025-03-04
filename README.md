# Crypto Price Prediction Project

## Problem Statement
This project aims to predict the next-day closing price of a cryptocurrency using historical data. The objective is to showcase data preprocessing, feature engineering, and modeling skills through the use of classical machine learning algorithms (Random Forest and XGBoost) as well as an LSTM-based deep learning model.

## Approach
1. **Data Preprocessing & Feature Engineering:**  
   The project begins by reading raw crypto data from a CSV file and cleaning it. The data is preprocessed by converting time columns, handling missing values, and ensuring all numeric fields are in the proper format. Next, technical indicators are generated using the `pandas_ta` library. These include multiple Exponential Moving Averages (EMAs) for various periods, Relative Strength Index (RSI), MACD (with its histogram and signal), Bollinger Bands, Stochastic Oscillator, and Average True Range (ATR). A target variable, "Next_Close," is also created by shifting the closing price by one period.

2. **Data Splitting:**  
   The enriched dataset is split into training and testing sets in chronological order (80% training, 20% testing) to respect the time series nature of the data. Features are selected efficiently without using iterative loops.

3. **Model Training & Evaluation:**  
   Three different modeling approaches are used:
   - **Random Forest Regressor:** A baseline ensemble model that captures non-linear patterns.
   - **XGBoost Regressor:** A gradient boosting model known for its competitive performance.
   - **LSTM Neural Network:** A deep learning model designed to capture sequential dependencies in time series data.  
   Each model is trained on the training data and evaluated on the test set using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

4. **Iterative Improvements & Experimentation:**  
   Although the initial models provide a good baseline, the project is structured to allow for further improvements via hyperparameter tuning, additional feature engineering, or experimentation with more advanced methods. This iterative approach demonstrates a clear pathway for future enhancements.

## Tools
- **Python:** The main programming language.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **Matplotlib:** For data visualization.
- **pandas_ta:** For calculating technical indicators.
- **Scikit-Learn:** For model building and evaluation (Random Forest, metrics, etc.).
- **XGBoost:** For gradient boosting regression.
- **TensorFlow/Keras:** For building and training the LSTM neural network.
- **Jupyter/Google Colab:** For an interactive coding environment.


Below is the converted text in Markdown format:

markdown
Copy
Edit
## Project Overview

Crypto Price Prediction is designed to predict the next closing price of a cryptocurrency by analyzing historical market data enriched with technical indicators. The project utilizes both classical ML models (Random Forest and XGBoost) and a deep learning approach (LSTM) to address the challenge, showcasing the versatility of modern AI techniques.

## Installation

**Clone the repository:**
   ```
   git clone https://github.com/kabradhruv/crypto-price-prediction.git
   ```

**Navigate to the project directory:**
```
cd crypto-price-prediction
```
**Install the required dependencies using pip:**
```
pip install -r requirements.txt
```
**Download the dataset :**
and place the CSV file in the root directory, naming it "crypto_data.csv".

**Usage**
Run the main Python script:
```
python ml_model_crypto.py
```

**The script will:**
- Preprocess the dataset and add technical features.
- Split the data into training and testing sets (80/20 split by time).
- Train and evaluate a Random Forest regressor and an XGBoost regressor.
- Create sequences and train an LSTM model.
- Model performance is printed to the console with metrics such as Mean Squared Error (MSE), Mean Absolute - Error (MAE), and R-squared (R²).

**Project Structure**
- ml_model_crypto.py: Main Python file containing all code for preprocessing, feature engineering, model training, and evaluation.
- README.txt: This documentation file.
- requirements.txt: List of Python packages required for the project.
- crypto_data.csv: Dataset file containing historical cryptocurrency data.
- Source Code
The complete source code is hosted on GitHub:
https://github.com/kabradhruv/crypto-price-prediction

**Additional Notes:**

The primary goal of this project is to demonstrate a clear, reproducible approach to solving a crypto prediction problem using both ML and AI techniques.
While the initial results may serve as a baseline, there is ample scope for improvement through hyperparameter tuning, additional feature engineering, and exploration of other advanced models.
The project structure and code are organized to facilitate ease of understanding and future modifications.
Conclusion
This project showcases a comprehensive workflow for cryptocurrency price prediction. From thorough data preprocessing and feature engineering to model training with both classical machine learning and deep learning methods, the project is a practical demonstration of applying Python, ML, and AI skills to a real-world problem. The clear documentation and modular code design ensure that the approach is understandable and reproducible.
