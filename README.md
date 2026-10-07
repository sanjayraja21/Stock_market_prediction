# Stock Market Prediction

## Project Overview

This project uses machine learning techniques to predict the **next trading day's closing price** of stocks based on historical stock market data.

The project compares three approaches:

* Lasso Regression
* XGBoost Regression
* Ensemble Model

The notebook performs data loading, data inspection, preprocessing, feature preparation, chronological train-test splitting, model training, prediction, evaluation, and visualization.

> **Note:** This project is for machine learning and educational purposes. Stock market predictions are not guaranteed to represent actual future market prices.

---

## Objective

The main objective of this project is to build and compare machine learning regression models for predicting the next trading day's closing price.

The target variable is:

```text
Target_Next_Close
```

It represents the next available trading day's closing price for each ticker.

---

## Dataset

The project uses:

```text
Merged_Stock_Market_Dataset.csv
```

The dataset contains:

* **177,728 rows**
* **10 columns**
* Historical data ranging from **2002-08-12 to 2026-09-17**

### Dataset Columns

| Column    | Description                             |
| --------- | --------------------------------------- |
| Date      | Trading date                            |
| Ticker    | Stock ticker symbol                     |
| Company   | Company/ticker identifier               |
| Open      | Opening price                           |
| High      | Highest price during the trading period |
| Low       | Lowest price during the trading period  |
| Close     | Closing price                           |
| Adj Close | Adjusted closing price                  |
| Volume    | Trading volume                          |
| Source    | Dataset source                          |

The notebook records the source as:

```text
Yahoo Finance Worldwide
```

---

## Features

The machine learning models use six numerical input features:

```text
Open
High
Low
Close
Adj Close
Volume
```

The target is:

```text
Target_Next_Close
```

The target is created by shifting the `Close` value by one trading day within each ticker:

```python
df["Target_Next_Close"] = (
    df.groupby("Ticker")["Close"].shift(-1)
)
```

---

## Data Preprocessing

The notebook performs the following preprocessing steps:

1. Loads the CSV dataset.
2. Inspects the dataset structure.
3. Checks missing values.
4. Checks duplicate rows.
5. Removes unnecessary whitespace from column names.
6. Converts the `Date` column to datetime.
7. Removes rows with missing dates.
8. Sorts the data by `Ticker` and `Date`.
9. Converts numerical columns into numeric data types.
10. Creates the next-day closing-price target.

The initial dataset inspection shows:

* No missing values.
* No duplicate rows.
* 177,728 records.
* 10 columns.

---

## Train-Test Split

Because this is time-series-related stock data, the notebook uses a **chronological 80/20 split** rather than a random split.

The 80% date point is used as the split date.

```text
Training data:
Date < split date

Testing data:
Date >= split date
```

This helps preserve the chronological order of the stock data.

---

## Machine Learning Models

### 1. Lasso Regression

Lasso Regression is used as a linear regression model with regularization.

The notebook uses:

```python
Lasso(
    alpha=0.001,
    max_iter=10000,
    random_state=42
)
```

Feature scaling is performed using:

```python
StandardScaler()
```

before training the Lasso model.

---

### 2. XGBoost Regression

The project also uses XGBoost Regression to capture more complex relationships between the input features and the target price.

The notebook uses an `XGBRegressor` model with multiple boosting trees and a learning rate of `0.05`.

---

### 3. Ensemble Model

The notebook also evaluates an ensemble prediction combining the model predictions.

This provides another approach for comparing the performance of the individual models.

---

## Evaluation Metrics

The models are evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

The square root of MSE.

### R² Score

Measures how well the model explains the variation in the target variable.

### Mean Absolute Percentage Error (MAPE)

Measures prediction error as a percentage.

---

## Model Results

The recorded notebook results are:

| Model     |         MAE |             MSE |         RMSE |         R² |   MAPE (%) |
| --------- | ----------: | --------------: | -----------: | ---------: | ---------: |
| **Lasso** | **29.3893** | **28,662.1298** | **169.2989** | **0.9995** | **1.4519** |
| XGBoost   |    252.9612 |  2,257,999.0394 |   1,502.6640 |     0.9602 |     2.9381 |
| Ensemble  |    109.0081 |    377,966.6430 |     614.7899 |     0.9933 |     1.9560 |

### Best Recorded Model

Based on the notebook's recorded test metrics, **Lasso Regression** achieved the best overall results among the three evaluated approaches.

```text
R²   = 0.9995
MAE  = 29.3893
RMSE = 169.2989
MAPE = 1.4519%
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Google Colab
* Jupyter Notebook

---

## Project Workflow

```text
Historical Stock Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Data Cleaning
        ↓
Date Processing
        ↓
Sort by Ticker and Date
        ↓
Create Next-Day Close Target
        ↓
Select Features
        ↓
Chronological 80/20 Split
        ↓
Feature Scaling for Lasso
        ↓
Model Training
   ┌────┴─────┐
   ↓          ↓
 Lasso     XGBoost
   └────┬─────┘
        ↓
   Ensemble Model
        ↓
   Predictions
        ↓
 Model Evaluation
        ↓
 Performance Comparison
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Stock_market_prediction.git
```

### 2. Open the notebook

Open:

```text
Stock_Market_Prediction.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Provide the dataset

The notebook uses the Google Colab file upload functionality.

Upload:

```text
Merged_Stock_Market_Dataset.csv
```

when prompted.

### 5. Run the notebook

Run the cells from top to bottom.

---

## Requirements

The project requires Python and the following libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

---

## Important Note

This project demonstrates machine learning techniques for historical stock-price prediction.

A high evaluation score on historical test data does **not** guarantee accurate future stock-market predictions. Real-world stock prices can be affected by many factors that are not included in this project, such as news, economic conditions, company announcements, market sentiment, and unexpected events.

This project should therefore be considered an educational machine learning project rather than financial advice.

---

## Author

**Sanjay R.**

B.Tech Artificial Intelligence and Data Science

---

## License

This project is intended for educational and portfolio purposes.
