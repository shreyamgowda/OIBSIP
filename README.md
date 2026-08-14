Car Price Prediction — Complete Process

1. Problem Definition

The objective of the task is to build a machine learning model that predicts the selling price of a used car based on its available characteristics.

The model learns relationships between car attributes such as vehicle age, kilometres driven, fuel type, seller type, and transmission type and the historical selling price.

⸻

2. Dataset Loading

The CarDekho dataset is loaded and examined to understand:

* Number of records
* Number of features
* Column names
* Data types
* Sample records
* Statistical information

The target variable for prediction is selling_price.

⸻

3. Data Cleaning

The dataset is checked for:

* Missing values
* Duplicate records
* Unnecessary columns

Duplicate records are removed so that repeated observations don’t influence the model.

⸻

4. Exploratory Data Analysis (EDA)

EDA is performed to understand the relationship between different car characteristics and selling price.

The analysis includes:

* Selling-price distribution — shows how car prices are distributed.
* Kilometres driven vs selling price — examines whether usage affects price.
* Fuel type vs selling price — compares prices across fuel types.
* Vehicle age vs selling price — examines depreciation as cars become older.
* Correlation heatmap — shows relationships between numerical variables.

These visualizations help understand patterns and potential factors affecting car prices.

⸻

5. Feature Selection

Relevant variables are selected as inputs for the machine-learning models.

In your final workflow, features include:

* Vehicle age
* Kilometres driven
* Fuel type
* Seller type
* Transmission type

The target variable is:

Selling price

⸻

6. Categorical Data Encoding

Machine-learning algorithms require numerical input.

Therefore, categorical variables such as:

* Fuel type
* Seller type
* Transmission type

are converted into numerical representations using one-hot encoding.

⸻

7. Train-Test Split

The dataset is divided into two parts:

* Training data — used to teach the models.
* Testing data — used to evaluate how well the models perform on unseen data.

Your workflow uses an 80:20 split.

⸻

8. Model Training

Three regression algorithms are trained so their performance can be compared:

Linear Regression

Used as a basic baseline model. It attempts to represent the relationship between the input features and selling price using a linear relationship.

Random Forest Regression

Uses multiple decision trees and combines their predictions. It can capture more complex relationships than Linear Regression.

Gradient Boosting Regression

Builds models sequentially, with each new model attempting to improve the errors made by previous models. It is useful for capturing complex patterns in the data.

⸻

9. Model Evaluation

The models are evaluated using three metrics:

MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.
Lower MAE is better.

RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.
Lower RMSE is better.

R² Score

Measures how well the model explains the variation in selling prices.
A value closer to 1 is better.

⸻

10. Model Comparison

The performance of all three models is placed into a comparison table and visualized.

The model with the best overall performance is selected based mainly on:

* Higher R²
* Lower MAE
* Lower RMSE

In your current results, Gradient Boosting was selected as the best model.

⸻

11. Feature Importance

Feature importance analysis is performed using the tree-based model.

This helps identify which input variables have the greatest influence on predicted car prices.

In your current analysis, kilometres driven was one of the most influential features, followed by factors such as transmission type and vehicle age.

This makes the machine-learning model more interpretable because you can explain which factors are contributing most to price prediction.

⸻

12. Actual vs Predicted Analysis

An actual-versus-predicted graph is created to visually evaluate the model.

* Actual selling price is shown on one axis.
* Predicted selling price is shown on the other.

The closer the predictions are to the actual values, the better the model’s predictive performance.

⸻

13. New Car Price Prediction

Finally, the trained best-performing model is used to predict the selling price of a new/example car based on its characteristics.

This demonstrates how the trained machine-learning model could be used in a real-world application.

⸻

Overall Workflow

CarDekho Dataset
↓
Data Inspection
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Selection
↓
Categorical Encoding
↓
Train/Test Split
↓
Train 3 Regression Models
↓
MAE + RMSE + R² Evaluation
↓
Model Comparison
↓
Select Best Model
↓
Feature Importance
↓
Actual vs Predicted Analysis
↓
New Car Price Prediction