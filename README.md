# California House Price Prediction

A Machine Learning project that predicts median house values in California using the California Housing dataset and Ridge Regression.

## Project Overview

This project explores housing data, performs Exploratory Data Analysis (EDA), trains regression models, compares their performance, and prepares a trained model for use in an interactive web application.

The application is built with Streamlit and is intended for educational and portfolio purposes.

## Objectives

- Explore and understand the California Housing dataset.
- Analyze feature distributions, correlations, and outliers.
- Train and evaluate regression models.
- Tune regularization parameters using GridSearchCV and 5-fold cross-validation.
- Select a final model using cross-validation performance.
- Build an interactive house price prediction interface.

## Dataset

The project uses the California Housing dataset.

**Target variable:** `MedHouseVal`

The target is measured in units of $100,000. For example, a target value of 2.0 corresponds to approximately $200,000.

The dataset's target values are capped at approximately $500,000, which limits predictions for higher-value properties.

### Features

- `MedInc` — Median income in the block group, expressed in units of $10,000.
- `HouseAge` — Median house age.
- `AveRooms` — Average rooms per household.
- `AveBedrms` — Average bedrooms per household.
- `Population` — Block group population.
- `AveOccup` — Average household occupancy.
- `Latitude` — Geographic latitude.
- `Longitude` — Geographic longitude.

## Machine Learning Workflow

1. Data loading and inspection
2. Exploratory Data Analysis
3. Correlation and outlier analysis
4. Train-test split
5. Baseline Linear Regression model
6. Ridge, Lasso, and ElasticNet evaluation
7. Hyperparameter tuning using GridSearchCV
8. Model comparison and selection
9. Final model evaluation
10. Preparation for Streamlit deployment

## Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression

## Final Model and Results

**Selected model:** Ridge Regression

Ridge was selected because it achieved the lowest cross-validation MAE among the tuned regularized models.

| Metric | Test Result |
|---|---:|
| MAE | 0.5331 |
| RMSE | 0.7453 |
| R² | 0.5761 |

Since the target is measured in units of $100,000, the test MAE corresponds to approximately $53,310.

R² of 0.5761 indicates that the model explains approximately 57.61% of the variation in target values. It does not mean the model has 57.61% prediction accuracy.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Joblib

## Project Structure

```text
california-house-price-prediction/
├── app.py
├── house_price_model.pkl
├── california_housing.csv
├── requirements.txt
├── README.md
└── House_Price_Prediction_MLR.ipynb
```

The notebook and dataset filenames in this structure are examples; use the exact filenames present in your repository.

## Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application allows users to enter housing features and receive a predicted median house value.

## Deployment

The application can be deployed through Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the application entry point.

## Limitations

- Predictions are estimates, not actual real estate quotes.
- The model explains only part of the variation in house values.
- The dataset target is capped at approximately $500,000.
- Predictions may be unreliable for inputs outside the data distribution.

## Future Improvements

- Improve model performance through additional experiments.
- Add input validation and clearer prediction explanations.
- Improve the application's visual design.
- Deploy and test the application online.

## Disclaimer

This project is intended for learning and portfolio demonstration. Predictions should not be used as the sole basis for financial or real estate decisions.
