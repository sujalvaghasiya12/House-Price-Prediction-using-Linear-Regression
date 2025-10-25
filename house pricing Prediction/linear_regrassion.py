# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
data = pd.DataFrame({
    'Area': [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500],
    'Price':[100000,150000,200000,250000,300000,350000,400000,450000,500000,550000]
})

print(data)

# Step 3: Split Data
X = data[['Area']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Visualize Results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('House Price Prediction')
plt.show()
