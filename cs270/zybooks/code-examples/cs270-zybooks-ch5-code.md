# CS 270 Zybooks Ch. 5 Code Examples

## Linear Regression in Python (5.1.1)

```py
import warnings
warnings.simplefilter("ignore") 
```
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
```
```py
# Load the dataset
rent_all = pd.read_csv("rent18.csv")
```
```py
# Keep subset of features, drop missing values
rent = rent_all[["price", "beds", "baths", "sqft", "room_in_apt"]].dropna()
rent.head()
```

Using simple linear regression to predict rental price from square footage:

```py
# Define input and output features for predicting price based on square footage
X = rent[["sqft"]]
y = rent[["price"]]
```
```py
# Initialize and fit simple linear regression model
simpLinModel = LinearRegression()
simpLinModel.fit(X, y)
```
```py
# Estimated intercept weight
simpLinModel.intercept_
```
```py
# Estimated weight for sqft feature
simpLinModel.coef_
```
```py
# Plot the data and fitted model

# Find predicted values
yPredicted = simpLinModel.predict(X)

# Plot
plt.scatter(X, y, color="#1f77b4")
plt.plot(X, yPredicted, color="#ff7f0e", linewidth=2)
plt.xlabel("Square footage", fontsize=14)
plt.ylabel("Price ($)", fontsize=14)
plt.show()
```
```py
# Predict the price of a 2,500 square foot rental
simpLinModel.predict([[2500]])
```

Using multiple linear regression to predict price from square footage & number of beedroms:

```py
# Define input and output features
X = rent[["sqft", "beds"]]
y = rent[["price"]]
```
```py
# Initialize and fit multiple regression model
multRegModel = LinearRegression()
multRegModel.fit(X, y)
```
```py
# Estimated intercept weight
multRegModel.intercept_
```
```py
# Estimated weights for sqft and beds features
multRegModel.coef_
```
```py
# Predict the price of a 2,500 square foot rental with 2 bedrooms
multRegModel.predict([[2500, 2]])
```
```py
# Plot data and fitted model

# Create grid for prediction surface
Xvals = np.linspace(min(rent["sqft"]), max(rent["sqft"]), 20)
Yvals = np.linspace(min(rent["beds"]), max(rent["beds"]), 20)
Xg, Yg = np.meshgrid(Xvals, Yvals)
Zvals = np.array(
    multRegModel.intercept_[0]
    + (Xg * multRegModel.coef_[0, 0] + Yg * multRegModel.coef_[0, 1])
)

# Plot data and surface
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
ax.grid()
ax.scatter(rent[["sqft"]], rent[["beds"]], rent[["price"]], color="#1f77b4")
ax.set_xlabel("Square footage", fontsize=14)
ax.set_ylabel("Bedrooms", fontsize=14)
ax.set_zlabel("Price ($)", fontsize=14)
ax.plot_surface(Xg, Yg, Zvals, alpha=0.25, color="grey")
plt.show()
```

Add code below to predict price from sq ftg, bedrooms, and bathrooms using a multiple regression model.

```py
X = rent[["sqft", "beds", "baths"]]
y = rent[["price"]]

regModel = LinearRegression()
regModelFit = regModel.fit(X, y)
```
```py
# weights w1, w2, ..., wp
regModel.coef_
```
```py
# weight w0
regModel.intercept_
```
Complete the code to predict the rental price for a rental with 3 bedrooms and 1 bathroom.
```py
prediction = regModelFit.predict([[3, 1]])
```

## 5.1.2: 1

The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

- Initialize a simple linear regression model.
- Fit the model to the given input and output features.

The code provided contains all imports, loads the dataset, creates input and output feature sets, and prints the intercept and weight of the linear regression model.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[["economy_gdp_per_capita"]]
y = happiness[["happiness_score"]]

# Initialize a simple linear regression model
# ---------------
SLRHappiness = LinearRegression()

# Fit a simple linear regression model
SLRHappiness.fit(X, y)
# ------------

# Estimated intercept weight
print(SLRHappiness.intercept_)

# Estimated weight for economy_gdp_per_capita feature
print(SLRHappiness.coef_)
```

## 5.1.2: 2

The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

- Print the intercept and weight for a simple linear regression model.

The code provided contains all imports, loads the dataset, creates input and output feature sets, and initializes and fits a simple linear regression model.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[["health_life_expectancy"]]
y = happiness[["happiness_score"]]

# Initialize a simple linear regression model
happinessSLR = LinearRegression()

# Fit a simple linear regression model
happinessSLR.fit(X,y)

# Estimated intercept weight
print(happinessSLR.intercept_)

# Estimated weight for health_life_expectancy feature
print(happinessSLR.coef_)
```

Output:

```
[3.04167872]
[[4.17397345]]
```

## 5.1.2: 3

The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

- Initialize a multiple linear regression model.
- Fit a multiple linear regression model to the data.
- Predict the happiness score for a country with `trust_government_corruption` of 0.2 and `health_life_expectancy` of 0.8.

The code provided contains all imports, loads the dataset, and creates input and output feature sets.

```py
# Import packages and functions
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[["trust_government_corruption", "health_life_expectancy"]]
y = happiness[["happiness_score"]]

# --------------
# Initialize a multiple linear regression model
happinessSLR = LinearRegression()

# Fit a multiple linear regression model
happinessSLR.fit(X, y)

# Predict the happiness score for a country with trust_government_corruption=0.2 and health_life_expectancy=0.8
print(happinessSLR.predict([[0.2, 0.8]]))
# --------------
```

Output:

```
[[6.37484089]]
```