SALES PREDICTION USING PYTHON 

1. PROJECT TITLE

Sales Prediction Using Python 


2. INTRODUCTION

Sales prediction is an important application of data science and machine learning. Businesses spend money on different advertising channels such as television, radio, and newspapers. Understanding how these advertising investments affect sales can help businesses make better marketing decisions.

The purpose of this project is to develop a machine-learning system that can predict sales based on advertising expenditure.

In this project, historical advertising data is analyzed to understand the relationship between TV, Radio, Newspaper advertising expenditure and Sales. Machine-learning regression algorithms are then trained using this historical data to predict sales.

The project demonstrates the complete process of a machine-learning project, starting from data loading and preprocessing and continuing through data analysis, visualization, model building, evaluation, comparison, and prediction.


3. OBJECTIVE OF THE PROJECT

The main objective of this project is to develop a machine-learning model that can predict sales using advertising expenditure.

The specific objectives of the project are:

• To understand the advertising dataset.
• To clean and prepare the data.
• To identify missing values and duplicate records.
• To perform exploratory data analysis.
• To understand the relationship between advertising channels and sales.
• To visualize important relationships using graphs.
• To build different regression models.
• To evaluate the performance of the models.
• To compare the models and identify the best-performing model.
• To identify the most important advertising channel.
• To predict sales for new advertising expenditure.


4. DATASET

The project uses a dataset named Advertising.csv.

The dataset contains information about advertising expenditure and sales.

The main variables in the dataset are:

TV:
Amount spent on TV advertising.

Radio:
Amount spent on Radio advertising.

Newspaper:
Amount spent on Newspaper advertising.

Sales:
Sales generated from the advertising expenditure.

The three advertising variables are used as input features, while Sales is used as the target variable.

Therefore, the machine-learning problem can be represented as:

TV + Radio + Newspaper → Sales


5. TECHNOLOGIES USED

The project was developed using Python.

The main libraries used in the project are:

Pandas:
Used for loading, cleaning, manipulating, and analyzing the dataset.

NumPy:
Used for numerical calculations and mathematical operations.

Matplotlib:
Used for creating graphs and visualizations.

Seaborn:
Used for statistical visualization such as scatter plots, histograms, and heatmaps.

Scikit-learn:
Used for machine-learning algorithms, train-test splitting, predictions, and model evaluation.


6. DATA LOADING

The first step of the project was to load the Advertising.csv dataset into Python.

After loading the dataset, the first few records were displayed to understand its structure.

The dataset was also examined to identify the number of rows and columns, column names, data types, missing values, duplicate records, and statistical information.

This initial analysis helped ensure that the dataset was loaded correctly and was suitable for further processing.


7. DATA UNDERSTANDING

The dataset was analyzed to understand its structure and characteristics.

The following information was checked:

• Number of rows
• Number of columns
• Column names
• Data types
• Missing values
• Duplicate records
• Statistical summary

The first few rows of the dataset were also examined.

This step is important because understanding the dataset before applying machine-learning algorithms helps avoid errors and ensures that the correct variables are selected.


8. DATA CLEANING AND PREPROCESSING

Before building the machine-learning models, the dataset was cleaned and prepared.

The column names were checked and standardized to ensure that they could be used correctly during analysis.

An unnecessary index column present in the original dataset was removed because it does not provide useful information for predicting sales.

Missing values were checked to ensure that the dataset did not contain problematic empty records.

Duplicate records were also checked and removed where necessary.

After preprocessing, the required variables were:

TV
Radio
Newspaper
Sales

The cleaned dataset was then used for exploratory analysis and machine learning.


9. EXPLORATORY DATA ANALYSIS

Exploratory Data Analysis, also known as EDA, was performed to understand patterns and relationships in the dataset.

Different visualizations were created to analyze the data.

The main visualizations included:

• Sales distribution
• TV advertising versus Sales
• Radio advertising versus Sales
• Newspaper advertising versus Sales
• Correlation heatmap

These visualizations helped understand how different advertising channels are related to sales.


10. SALES DISTRIBUTION

A histogram was created to understand the distribution of sales values.

The histogram helps identify how the sales values are distributed throughout the dataset.

It provides an overview of the range of sales and shows where most of the observations are concentrated.

This analysis helps in understanding the general behavior of the target variable.


11. TV ADVERTISING VS SALES

A scatter plot was created to analyze the relationship between TV advertising expenditure and sales.

The graph helps determine whether changes in TV advertising expenditure are associated with changes in sales.

The relationship observed in the dataset indicates that TV advertising is an important variable for predicting sales.

The visualization provides an easy way to understand the relationship between TV advertising and sales.


12. RADIO ADVERTISING VS SALES

A scatter plot was created between Radio advertising expenditure and Sales.

This visualization helps determine whether higher radio advertising expenditure is associated with higher sales.

The relationship between Radio advertising and Sales was analyzed along with the other advertising channels.


13. NEWSPAPER ADVERTISING VS SALES

A scatter plot was created between Newspaper advertising expenditure and Sales.

This visualization helps determine the relationship between newspaper advertising expenditure and sales.

The results were compared with the TV and Radio relationships to understand the relative importance of each advertising channel.


14. CORRELATION ANALYSIS

Correlation analysis was performed to understand the strength and direction of relationships between the numerical variables.

Correlation values help determine whether two variables have a positive or negative relationship.

A positive correlation indicates that two variables tend to increase together, while a negative correlation indicates that one variable tends to decrease as the other increases.

The correlation analysis helped identify which advertising channels have stronger relationships with sales.


15. CORRELATION HEATMAP

A correlation heatmap was created to visualize the relationships between the variables.

The heatmap makes it easier to identify strong and weak relationships between TV, Radio, Newspaper, and Sales.

This visualization provided a quick understanding of which advertising variables may be useful for predicting sales.


16. FEATURE SELECTION

After analyzing the dataset, the following variables were selected as input features:

TV
Radio
Newspaper

Sales was selected as the target variable.

The machine-learning models use the three advertising variables to predict Sales.

Therefore, the prediction process can be represented as:

Advertising Expenditure → Sales Prediction


17. TRAIN-TEST SPLIT

The dataset was divided into training and testing data.

Approximately 80% of the data was used for training the machine-learning models.

The remaining 20% was used for testing the models.

The training data was used by the models to learn patterns and relationships between advertising expenditure and sales.

The testing data was used to evaluate how well the models perform on unseen data.

This helps determine whether the models can generalize to new observations.


18. MACHINE-LEARNING MODELS

Three regression models were developed and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Using multiple models allows their performances to be compared and helps identify the most suitable model for the dataset.


19. LINEAR REGRESSION

Linear Regression was used as a baseline machine-learning model.

The model attempts to establish a mathematical relationship between the advertising variables and sales.

It learns how TV, Radio, and Newspaper advertising expenditure are associated with sales.

After training, the Linear Regression model was used to predict sales for the testing data.

The predictions were then evaluated using different performance metrics.


20. RANDOM FOREST REGRESSOR

Random Forest Regressor was used as the second machine-learning model.

Random Forest is an ensemble learning technique that uses multiple decision trees to produce predictions.

Instead of depending on a single decision tree, Random Forest combines the predictions from many trees.

This can improve prediction performance and make the model more robust.

The Random Forest model was trained using the advertising features and tested using unseen data.


21. GRADIENT BOOSTING REGRESSOR

Gradient Boosting Regressor was used as the third model.

Gradient Boosting builds models sequentially, where each new model attempts to improve the errors made by previous models.

This makes Gradient Boosting a powerful technique for regression problems.

The model was trained using the training dataset and evaluated using the testing dataset.


22. MODEL EVALUATION

The three machine-learning models were evaluated using four performance metrics:

Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted sales.

A lower MAE indicates better performance.

Mean Squared Error (MSE)

MSE calculates the average squared difference between actual and predicted sales.

A lower MSE indicates better performance.

Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

It represents the prediction error in approximately the same units as the target variable.

A lower RMSE indicates better performance.

R² Score

R² Score measures how well the model explains the variation in the target variable.

A higher R² Score generally indicates better model performance.


23. MODEL COMPARISON

The performance of Linear Regression, Random Forest, and Gradient Boosting was compared using MAE, MSE, RMSE, and R² Score.

The model with lower error values and a higher R² Score was considered the better-performing model.

A model comparison table was created in the project to compare the performance of all three algorithms.

This comparison helped identify the most suitable model for the sales prediction task.


24. FEATURE IMPORTANCE

Feature importance was analyzed using the Random Forest model.

The importance of the three advertising channels was compared:

• TV
• Radio
• Newspaper

Feature importance indicates how useful each feature is for making predictions.

The feature with the highest importance value was identified as the most influential advertising channel in the Random Forest model.

This analysis provides useful information about which advertising channel has the strongest predictive contribution to sales.


25. ACTUAL VS PREDICTED SALES

An Actual vs Predicted Sales graph was created to visually evaluate the performance of the model.

The actual sales values were compared with the predicted sales values.

If the predicted values are close to the actual values, the model is performing well.

This visualization provides an additional method of evaluating the quality of the predictions.


26. NEW SALES PREDICTION

After training the machine-learning models, new advertising expenditure values were provided to the trained model.

The new values included expenditure on:

• TV
• Radio
• Newspaper

The trained model then generated a predicted sales value.

This demonstrates how the developed model can be used to predict sales for new advertising scenarios.


27. BUSINESS APPLICATION

The project can be useful for businesses that want to understand the relationship between advertising expenditure and sales.

A business can use the model to estimate expected sales for different advertising budgets.

The project can support:

• Advertising budget planning
• Marketing strategy
• Sales forecasting
• Identification of important advertising channels
• Comparison of advertising investments
• Data-driven decision making

The model should be used as a decision-support tool along with other business information.


28. ADVANTAGES OF THE PROJECT

The main advantages of the project are:

• It uses real-world advertising data.
• It demonstrates a complete machine-learning workflow.
• It compares multiple regression algorithms.
• It provides measurable model performance.
• It identifies important advertising features.
• It can predict sales for new advertising expenditure.
• It demonstrates practical use of Python and machine learning.
• It can support marketing and business decisions.


29. LIMITATIONS

The dataset contains information about only three advertising channels: TV, Radio, and Newspaper.

Actual sales may also depend on several other factors, such as:

• Product price
• Competition
• Seasonal demand
• Promotions
• Customer preferences
• Economic conditions
• Brand awareness
• Distribution
• Market conditions

These factors are not included in the dataset.

Therefore, the model's predictions are based only on the variables available in the dataset.


30. FUTURE IMPROVEMENTS

The project can be improved in several ways.

More features could be added to improve the accuracy of the predictions.

Cross-validation could be used to obtain more reliable estimates of model performance.

Hyperparameter tuning could be performed to optimize the Random Forest and Gradient Boosting models.

Additional machine-learning algorithms could also be tested.

The final model could also be deployed as a web application where users enter advertising expenditure and receive predicted sales.


31. PROJECT WORKFLOW

The complete project workflow is:

Dataset Collection

↓

Data Loading

↓

Data Understanding

↓

Data Cleaning

↓

Missing Value and Duplicate Checking

↓

Exploratory Data Analysis

↓

Data Visualization

↓

Correlation Analysis

↓

Feature Selection

↓

Train-Test Split

↓

Linear Regression

↓

Random Forest

↓

Gradient Boosting

↓

Model Evaluation

↓

Model Comparison

↓

Feature Importance Analysis

↓

New Sales Prediction

↓

Final Conclusion


32. CONCLUSION

The Sales Prediction Using Python and Machine Learning project was successfully completed.

The project analyzed the relationship between advertising expenditure and sales using the Advertising.csv dataset.

The data was loaded, cleaned, and analyzed to understand its structure and characteristics.

Exploratory Data Analysis was performed using histograms, scatter plots, and a correlation heatmap to understand the relationships between advertising expenditure and sales.

Three regression models were developed: Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.

The models were evaluated using Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R² Score.

The performance of the models was compared to identify the best-performing model.

Feature importance analysis was performed to determine which advertising channel had the greatest predictive contribution.

Finally, the trained model was used to predict sales for new advertising expenditure values.

Overall, this project demonstrates the complete machine-learning workflow, including data preprocessing, exploratory data analysis, visualization, model development, model evaluation, feature analysis, model comparison, and prediction.

The project provided practical experience in using Python and machine-learning techniques to solve a real-world sales prediction problem.

PROJECT STATUS: COMPLETED SUCCESSFULLY