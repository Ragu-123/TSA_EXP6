### Name  : RAGUNATH R 
### Reg.No: 212222240081
### Date  : 

# Ex.No: 6               HOLT WINTERS METHOD
### AIM:
   To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. Load and resample the gold price data to monthly frequency, selecting the 'Daily_MInimum_Temperatures' column.
2. Scale the data using Minmaxscaler then split into training (80%) and testing (20%) sets.
3. Fit an additive Holt-Winters model to the training data and forecast on the test data.
4. Evaluate model performance using MAE and RMSE, and plot the train, test, and prediction results.
5. Train a final multiplicative Holt-Winters model on the full dataset and forecast future Temperatures.
### PROGRAM:
```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Load the dataset
data = pd.read_csv('/content/dataset.csv', index_col='Date', parse_dates=True)

# Convert the 'Daily minimum temperatures' column to numeric
data['Daily minimum temperatures'] = pd.to_numeric(data['Daily minimum temperatures'], errors='coerce')

# Resample the data to monthly averages for analysis
monthly_data = data.resample('MS').mean()

# Select the 'Daily minimum temperatures' column for analysis
data_column = monthly_data['Daily minimum temperatures']

# Remove any remaining NaN values
data_column = data_column.dropna()

# Scaling the Data using MinMaxScaler 
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data_column.values.reshape(-1, 1)).flatten(), index=data_column.index)

# Split into training and testing sets (80% train, 20% test)
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

# Fit an additive Exponential Smoothing model
fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
mae = mean_absolute_error(test_data, test_predictions_add)
rmse = mean_squared_error(test_data, test_predictions_add, squared=False)

# Print performance metrics
print("MAE:", mae)
print("RMSE:", rmse)

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()

# Fit a multiplicative model to forecast future values
final_model = ExponentialSmoothing(data_column, trend='mul', seasonal='mul', seasonal_periods=12).fit()

# Forecast future values for the next 12 months
forecast_predictions = final_model.forecast(steps=12)

# Plot current data and future predictions
data_column.plot(figsize=(12, 8), legend=True, label='Current Temperature')
forecast_predictions.plot(legend=True, label='Forecasted Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Forecast')
plt.show()
```

### OUTPUT:

![WhatsApp Image 2024-10-16 at 13 29 27_d7403d9a](https://github.com/user-attachments/assets/b41a3347-812f-42f1-878d-6ffe3cff9c4f)


#### TEST_PREDICTION
![WhatsApp Image 2024-10-16 at 13 20 02_29ef66e1](https://github.com/user-attachments/assets/6be56ff4-0899-4aa7-b2c7-9c1ca6013730)

#### FINAL_PREDICTION
![WhatsApp Image 2024-10-16 at 13 20 54_7aeeeaec](https://github.com/user-attachments/assets/64fcf6fc-b5b9-440e-a315-ba7efc5dade1)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
