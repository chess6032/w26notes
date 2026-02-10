# CS 270 Zybooks Ch. 5 Code Examples

## 5.1

### Linear Regression in Python (5.1.1)

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

### 5.1.2: 1

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

### 5.1.2: 2

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

### 5.1.2: 3

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

## 5.2

### 5.2.1: Elastic net regression using sklearn

The following Python code loads the full dataset for rentals listed in 2018 and fits linear regression models for predicting rental price using elastic net regression.

<!-- - Modify the regularization strength to `alpha=5` in the simple linear regression model predicting price from square footage. Re-run the code and examine changes in estimated weights.
- Modify the weight applied to the L1 regularization term to `l1_ratio=0.75` in the multiple linear regression model predicting price from square footage and bedrooms. Re-run the code and examine changes in estimated weights. -->

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
```

```py
# Load the dataset
rent_all = pd.read_csv("rent18.csv")

# Keep subset of features, drop missing values
rent = rent_all[["price", "beds", "baths", "sqft"]].dropna()
rent.head()
```

Use elastic net regression to predict rental price from square footage

```py
# Define input and output features for predicting price from sqft
X = rent[["sqft"]]
y = rent[["price"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
```py
# Initialize and fit model using elastic net regression
eNet = ElasticNet(alpha=1.0, l1_ratio=0.5)
eNet.fit(X, y)
```
```py
# Estimated intercept weight
eNet.intercept_
```
```py
# Estimated weight for sqft
eNet.coef_
```

Compare elastic net to least squares

```py
# Fit using least squares

linRegModel = LinearRegression()
linRegModel.fit(X, y)
```
```py
linRegModel.intercept_
```
```py
linRegModel.coef_
```
```py
# Plot the data and both fitted models

# Find predicted values
yPredictedENet = eNet.predict(X)
yPredictedLin = linRegModel.predict(X)

# Plot
plt.scatter(X, y, color="#1f77b4", s=10)
plt.plot(X, yPredictedENet, color="#ff7f0e", linewidth=2, label="Elastic net")
plt.plot(X, yPredictedLin, color="#3ca02c", linewidth=2, label="Least squares")
plt.xlabel("Standardized square footage", fontsize=14)
plt.ylabel("Price ($)", fontsize=14)
plt.legend(loc="upper left")
plt.show()
```

Use elastic net regression to predict rental price from square fottage and number of bedrooms.

```py
# Define input and output features
X = rent[["sqft", "beds"]]
y = rent[["price"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
```py
# Initialize and fit elastic net
eNet2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
eNet2.fit(X, y)
```
```py
# Estimated intercept and weights
print(eNet2.intercept_)
print(eNet2.coef_)
```
```py
# Plot the absolute value of the weights
importance = np.abs(eNet2.coef_)
names = np.array(["sqft", "beds"])
sort = np.argsort(importance)[::-1]
plt.bar(x=names[sort], height=importance[sort])
plt.ylabel("Importance", fontsize=14)
plt.show()
```

Set $\alpha=1$, show weight estimates for different values of the weight applied to the L1 norm, $\lambda$.

```py
aVal = 1
l1vals = np.linspace(
    0.01,
    1,
    100,
)
ENcoef = np.empty([100, 3])

for i in range(len(l1vals)):
    EN = ElasticNet(alpha=aVal, l1_ratio=l1vals[i], max_iter=10000)
    EN.fit(X, y)
    ENcoef[i, 0] = EN.intercept_[0]
    ENcoef[i, 1] = EN.coef_[0]
    ENcoef[i, 2] = EN.coef_[1]

# Plot
fig = plt.figure(figsize=(5.5, 4))
plt.plot(
    l1vals,
    ENcoef[:, 1],
    color="#1f77b4",
    linestyle="solid",
    linewidth=3,
    label=r"$w_1$",
)
plt.plot(
    l1vals,
    ENcoef[:, 2],
    color="#ff7f0e",
    linestyle="dashed",
    linewidth=3,
    label=r"$w_2$",
)

plt.xlabel(r"Weight applied to L1 norm, $\lambda$", fontsize=14)
plt.ylabel("Estimate", fontsize=14)

plt.legend(loc="upper left", fontsize=14)

plt.show()
```

Set the weight applied to the L1 norm at $\lambda = 0.5$, show weight estimates for different values of the regularization strength, $\alpha$.

```py
l1NormWeight = 0.5
alphaVals = np.logspace(-1, 2, 100)
ENcoef = np.empty([100, 3])

for i in range(len(ENcoef)):
    EN = ElasticNet(alpha=alphaVals[i], l1_ratio=l1NormWeight, max_iter=2000)
    EN.fit(X, y)
    ENcoef[i, 0] = EN.intercept_[0]
    ENcoef[i, 1] = EN.coef_[0]
    ENcoef[i, 2] = EN.coef_[1]


# Plot
fig = plt.figure(figsize=(6, 4))
plt.plot(
    alphaVals,
    ENcoef[:, 1],
    color="#1f77b4",
    linestyle="solid",
    linewidth=3,
    label=r"$w_1$",
)
plt.plot(
    alphaVals,
    ENcoef[:, 2],
    color="#ff7f0e",
    linestyle="dashed",
    linewidth=3,
    label=r"$w_2$",
)


plt.xlabel(r"Regularization strength, $\alpha$", fontsize=14)
plt.ylabel("Estimate", fontsize=14)

plt.legend(loc="upper right", fontsize=14)

plt.show()
```

### 5.2.2: 1

The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

- Initialize an elastic net regression model.
- Fit the model to the given input and output features.

The code provided contains all imports, loads the dataset, creates input and output feature sets, standardizes the input features, and prints the intercept and weight of the elastic net regression model.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[["economy_gdp_per_capita"]]
y = happiness[["happiness_score"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------------------
# Initialize an elastic net regression model
eNetHappiness = ElasticNet()

# Fit the model 
eNetHappiness.fit(X, y)
# ---------------------------

# Print estimated intercept weight
print(eNetHappiness.intercept_)

# Print estimated weight for economy_gdp_per_capita feature
print(eNetHappiness.coef_)
```

Output:

    [5.40968]
    [0.30094963]

### 5.2.2: 2

The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

- Initialize an elastic net regression model with `alpha=43` and `l1_ratio=0.82`.

The code provided contains all imports, loads the dataset, creates input and output feature sets, standardizes the input features, fits the elastic net regression model, and prints the intercept and weight.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[["economy_gdp_per_capita", "generosity"]]
y = happiness[["happiness_score"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize an elastic net regression model with alpha=43 and l1_ratio=0.82
# ---------------------
happinessENet = ElasticNet(alpha=43, l1_ratio=0.82)
# ---------------------

# Fit the model 
happinessENet.fit(X,y)

# Print estimated intercept weight
print(happinessENet.intercept_)

# Print estimated weights for input features
print(happinessENet.coef_)
```

Output:

    [5.40535999]
    [0. 0.]


### 5.2.2: 3

The World Happiness Report ranks 155 countries based on data from the Gallup World Poll. People completing the survey are asked to rate their current lives, which is averaged across countries to produce a rating on a scale from 0 (worst possible) to 10 (best possible).

- Print the intercept and weights for an elastic net regression model.
- Predict the happiness score for a country with standardized `freedom` of -1.96 and `economy_gdp_per_capita` of -1.91.

The code provided contains all imports, loads the dataset, creates input and output feature sets, standardizes the input features, and initializes and fits an elastic net regression model.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Load the world happiness dataset
happiness = pd.read_csv("world_happiness_2017.csv")

# Define input and output features
X = happiness[["freedom", "economy_gdp_per_capita"]]
y = happiness[["happiness_score"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize an elastic net regression model
modelENet = ElasticNet()

# Fit the model 
modelENet.fit(X,y)

# -------------------
# Print estimated intercept weight
print(modelENet.intercept_)

# Print estimated weights for input features
print(modelENet.coef_)

# Predict and print the happiness score for a country with standardized
# freedom=-1.96 and economy_gdp_per_capita=-1.91
print(modelENet.predict([[-1.96, -1.91]]))
# -------------------
```

Output:

    [5.31562]
    [0.04920547 0.29733263]
    [4.65127195]



## 5.3: KNN Regression

### 5.3.1: kNN regression using sklearn

The following Python code loads the full dataset for rentals listed in 2018. The price for rentals in San Jose is predicted using k-nearest neighbors regression from a single input feature, sqft, and then from three input features: sqft, baths, and beds. The price of a rental with 2000 square feet, 2 bedrooms, and 1 bathroom is predicted.

I did not do ts but here were the instructions:

- Modify the code predicting price from square footage to use k=15 nearest neighbors. Re-run the code and examine changes in the prediction line plotted with the data.
- Compare results from the k-nearest neighbors regression model fit using unstandardized input features and the model fit using standardized input features.
- Modify the code initializing the knnrScaled model to use Manhattan distance. Re-run the code and examine changes in the predicted price for the new instance and the instance's nearest neighbors.

```py
import warnings
warnings.simplefilter("ignore")
```
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
```
```py
# Load the dataset
rent_all = pd.read_csv("rent18.csv")

# Keep subset of features, drop missing values
rent = rent_all[rent_all["city"] == "san jose"]
rent = rent[["price", "beds", "baths", "sqft"]].dropna()
rent.head()
```
```py
# Predict price from sqft
# Define input and output features
X = rent[["sqft"]]
y = rent[["price"]]


# Plot sqft and price
fig = plt.figure(figsize=(4, 3))
plt.scatter(X, y, color="#1f77b4")
plt.xlabel("Square footage", fontsize=14)
plt.ylabel("Price ($)", fontsize=14)
plt.ylim([700, 5000])
plt.xlim([200, 3000])

plt.show()
```
```py
# Initiate and fit a k-nearest neighbors regression model with k=5
knnr = KNeighborsRegressor(n_neighbors=5)
knnrFit = knnr.fit(X, y)
```
```py
# Define a new instance with 2000 square feet
Xnew = [[2000]]

# Predict price for new instance
neighbors = knnrFit.predict(Xnew)
neighbors
```
```py
# Find the 5 nearest neighbors for the new instance
neighbors = knnrFit.kneighbors(Xnew)

# Return only the distances between the new instance and each of the the 5 nearest neighbors
neighbors[0]
```
```py
# Return the data frame instances of the 5 nearest neighbors
rent.iloc[neighbors[1][0]]
```

Instances in the neighborhood have similar square footage, but different numbers of bedrooms and bathrooms. Adding beds and baths as input features, in addition to square footage, seems reasonable.

```py
# Plot data with k-nearest neighbors prediction
Xvals = np.linspace(200, 3000, 100).reshape(-1, 1)
knnrPred = knnr.predict(Xvals)

fig = plt.figure(figsize=(4, 3))
plt.scatter(X, y, color="#1f77b4")
plt.plot(Xvals, knnrPred, color="#ff7f0e", linewidth=2)
plt.xlabel("Square footage", fontsize=14)
plt.ylabel("Price ($)", fontsize=14)
plt.ylim([700, 5000])
plt.xlim([200, 3000])

plt.show()
```
```py
# Define input features as sqft, beds, and baths
X = rent[["sqft", "beds", "baths"]]
y = rent[["price"]]

# Scale the input features
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)
```
```py
# Initiate and fit a k-nearest neighbors regression model with k=5 on unscaled input features
knnrUnscaled = KNeighborsRegressor(n_neighbors=5)
knnrUnscaledFit = knnrUnscaled.fit(X, y)
```
```py
# Initiate and fit a k-nearest neighbors regression model with k=5 on unscaled input features
knnrScaled = KNeighborsRegressor(n_neighbors=5)
knnrScaledFit = knnrScaled.fit(Xscaled, y)
```
```py
# Define new instance with 2000 square feet, 2 bedrooms, 1 bathroom
Xsqft = 2000
Xbeds = 2
Xbaths = 1
Xnew = [[Xsqft, Xbeds, Xbaths]]
Xnew
```
```py
# Predict price for new instance using unscaled input features
print("Prediction from unscaled input features: ", knnrUnscaledFit.predict(Xnew)[0][0])

# Predict price for new instance using scaled input features
# Find scaled input features for new instance
XsqftScaled = (Xsqft - rent["sqft"].mean()) / (rent["sqft"].var() ** 0.5)
XbedsScaled = (Xbeds - rent["beds"].mean()) / (rent["beds"].var() ** 0.5)
XbathsScaled = (Xbaths - rent["baths"].mean()) / (rent["baths"].var() ** 0.5)

XnewScaled = [[XsqftScaled, XbedsScaled, XbathsScaled]]

print(
    "Prediction from scaled input features: ", knnrScaledFit.predict(XnewScaled)[0][0]
)
```

    Prediction from unscaled input features: 3740.0
    Prediction from scaled input features: 2609.0

```py
# Unscaled nearest neighbors
rent.iloc[knnrUnscaledFit.kneighbors(Xnew)[1][0]]
```
```py
# Scaled nearest neighbors
rent.iloc[knnrScaledFit.kneighbors(XnewScaled)[1][0]]
```