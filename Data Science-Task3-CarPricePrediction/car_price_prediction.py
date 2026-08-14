# %%
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = pd.read_csv("cardekho_dataset.csv")
print(data.columns.tolist())
print("first 5 rows:")
print(data.head())
print("\nDataset Information:")
print(data.info())
print("\nnumber of rows and columns:")
print(data.shape)
print("\nmissing values:")
print(data.isnull().sum())
print("column names:")
print(data.columns)
print("\nSummary Statistics:")
print(data.describe())
print("duplicate rows:",data.duplicated().sum())
data=data.drop_duplicates()
print("dataset shape after removing duplicates:")
print(data.shape)
print(data.isnull().sum())
plt.figure(figsize=(8,5))
sns.histplot(data['selling_price'],bins=30,kde=True)
plt.title("distribution of selling prices")
plt.xlabel("selling price")
plt.ylabel("Number of cars")
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(x='km_driven',y='selling_price',data=data)
plt.title("kilometers driven vs selling price")
plt.xlabel("kilometers driven")
plt.ylabel("selling price")
plt.show()
x=data[['vehicle_age','km_driven','fuel_type','seller_type','transmission_type']]
plt.figure(figsize=(8,5))
sns.boxplot(x='fuel_type',y='selling_price',data=data)
plt.title('selling price by fuel type')
plt.xlabel('fuel type')
plt.ylabel('selling price')
plt.show()
# Price vs Car Age
plt.figure(figsize=(8, 5))
sns.scatterplot(x='vehicle_age', y='selling_price', data=data)
plt.title('Selling Price vs Car Age')
plt.xlabel('Car Age (Years)')
plt.ylabel('Selling Price')
plt.show()
# Feature Correlation Heatmap
numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_data.corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
y=data['selling_price']
x=pd.get_dummies(x,drop_first=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
print("Model Trained Succesfully!")
y_pred=model.predict(x_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
print("\nmodel evaluation")
print("-------------")
print("mean absolute error (MAE):",mae)
print("mean squared error(MSE):",mse)
print("Root Mean Squared Error(RMSE):",rmse)
print("R2 Score:",r2)
rf_model = RandomForestRegressor(n_estimators=200,random_state=42,max_depth=15)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)
print("\nRandom Forest Model Evaluation")
print("--------------------------------")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)
gb_model = GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,max_depth=3,random_state=42)
gb_model.fit(x_train, y_train)
gb_pred = gb_model.predict(x_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_rmse = np.sqrt(gb_mse)
gb_r2 = r2_score(y_test, gb_pred)
print("\nGradient Boosting Model Evaluation")
print("-----------------------------------")
print("MAE:", gb_mae)
print("MSE:", gb_mse)
print("RMSE:", gb_rmse)
print("R2 Score:", gb_r2)
model_results = pd.DataFrame({"Model": ["Linear Regression","Random Forest","Gradient Boosting"],"MAE": [mae,rf_mae,gb_mae],"RMSE": [rmse,rf_rmse,gb_rmse],"R2 Score": [r2,rf_r2,gb_r2]})
print("\nModel Comparison")
print("----------------")
print(model_results)
plt.figure(figsize=(8,5))
sns.barplot(x="Model",y="R2 Score",data=model_results)
plt.title("R2 Score Comparison of Regression Models")
plt.xlabel("Regression Model")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.show()
best_model_row = model_results.loc[
    model_results["R2 Score"].idxmax()]
print("\nBest Model:")
print(best_model_row)
importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({"Feature": x_train.columns,"Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance",ascending=False)
print("\nFeature Importance:")
print(feature_importance)
plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
sample_car = x_test.iloc[[0]]
predicted_price = rf_model.predict(sample_car)
print("Predicted Car Price:", predicted_price[0])
print("Actual Car Price:", y_test.iloc[0])
data = data.drop(columns=["Unnamed: 0"], errors="ignore")
data = pd.read_csv("cardekho_dataset.csv")
data = data.drop(columns=["Unnamed: 0"], errors="ignore")
print(data.columns.tolist())
data["vehicle_age"] = 2026 - data["vehicle_age"]
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
linear_pred = linear_model.predict(x_test)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)
# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
gb_model.fit(x_train, y_train)
gb_pred = gb_model.predict(x_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)
print("\n================ MODEL COMPARISON ================")
print("\nLinear Regression")
print("MAE :", linear_mae)
print("RMSE:", linear_rmse)
print("R2  :", linear_r2)
print("\nRandom Forest")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)
print("\nGradient Boosting")
print("MAE :", gb_mae)
print("RMSE:", gb_rmse)
print("R2  :", gb_r2)
results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Random Forest',
        'Gradient Boosting'
    ],
    'MAE': [
        linear_mae,
        rf_mae,
        gb_mae
    ],
    'RMSE': [
        linear_rmse,
        rf_rmse,
        gb_rmse
    ],'R2 Score': 
    [linear_r2,
     rf_r2,
     gb_r2
     ]
     })
print("\nFINAL MODEL COMPARISON")
print("=" * 70)
print(results)
plt.figure(figsize=(8, 5))
plt.bar(
    results["Model"],
    results["R2 Score"]
)
plt.title("R2 Score Comparison of Models")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.show()
best_model_row = results.loc[results["R2 Score"].idxmax()]
print("\n================ BEST MODEL ================")
print("Best Model:", best_model_row["Model"])
print("MAE:", best_model_row["MAE"])
print("RMSE:", best_model_row["RMSE"])
print("R2 Score:", best_model_row["R2 Score"])
feature_importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print("\n================ FEATURE IMPORTANCE ================")
print(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],"MAE": [linear_mae,rf_mae,gb_mae],
    "RMSE": [linear_rmse,rf_rmse,gb_rmse],"R2 Score": [linear_r2,rf_r2,gb_r2]})
print("\n========== FINAL MODEL COMPARISON ==========")
print(results)
# Select the best model based on R2 Score
best_model_name = results.loc[results["R2 Score"].idxmax(), "Model"]
print("\n================================")
print("BEST MODEL:", best_model_name)
print("================================")
# Choose the best trained model
if best_model_name == "Linear Regression":
    final_model = linear_model
elif best_model_name == "Random Forest":
    final_model = rf_model
else:
    final_model = gb_model
# Create a sample new car
new_car = pd.DataFrame({
    "vehicle_age": [5],
    "km_driven": [50000],
    "seller_type": ["Individual"],
    "fuel_type": ["Petrol"],
    "transmission_type": ["Manual"]})
# Convert categorical variables to dummy variables
new_car = pd.get_dummies(new_car, drop_first=True)
# Make sure new_car has exactly the same columns as training data
new_car = new_car.reindex(columns=x_train.columns, fill_value=0)
# Predict price
predicted_price = final_model.predict(new_car)
print("\n========== NEW CAR PREDICTION ==========")
print("Vehicle Age : 5 years")
print("KM Driven   : 50,000 km")
print("Seller Type : Individual")
print("Fuel Type   : Petrol")
print("Transmission: Manual")
print("----------------------------------------")
print("Predicted Car Price: ₹", round(predicted_price[0],2))