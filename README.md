
# California Housing Price Prediction

## Project Overview

This project focuses on analyzing California housing data
and building machine learning regression models to predict
median house values.

The main goal is to understand the dataset, explore
relationships between features, train regression models,
and evaluate their performance.

## Dataset

California Housing Dataset from Scikit-learn.

The dataset contains 20,640 samples and 8 input features.

Target:
- MedHouseVal: Median house value (in units of 100,000 USD).

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Exploratory Data Analysis

The project includes:

- Dataset inspection
- Missing value analysis
- Duplicate value analysis
- Target distribution visualization
- Feature correlation analysis
- Income and house value visualization

## Machine Learning Models

The following regression models are implemented:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression

## Model Evaluation

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R2 Score

## Error Analysis

The project also examines prediction errors
using actual vs predicted plots and residual plots.

## Project Structure

- data/: Dataset information
- notebooks/: Data analysis and experiments
- src/: Training script
- results/: Model comparison results
- models/: Saved model files

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python src/train.py
```

## Results

Add the actual model evaluation results here
after running the project.

## Limitations

The dataset represents historical California
housing information.

Its target values have an upper limit, so the
model should not be interpreted as a predictor
of current market prices.

## Author

Amirali malekjafarian
