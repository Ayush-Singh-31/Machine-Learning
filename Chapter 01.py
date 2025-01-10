# The code is written in Python and uses the libraries numpy, pandas, matplotlib, and scikit-learn.
# The code is based on the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Importing the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat["GDP per capita (USD)"].values
y = lifesat["Life satisfaction"].values

# Plotting the data
lifesat.plot(kind='scatter', x="GDP per capita (USD)", y='Life satisfaction', grid=True)
plt.axis([23_500, 62_500,4,9])
plt.show()

# Training the Linear Regression model
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
X_new = [[37_655.2]]
print(model.predict(X_new))

# Training the K-Nearest Neighbors Regressor model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X.reshape(-1, 1), y)
X_new = [[37_655.2]]
print(model.predict(X_new))