TODO: Note down vocab from 1.1 & 1.2. ("Supervised models", "Unsupervised models", "features" vs. "instances", "dataset" & "dataframe", "Classification model" vs. "Regression model", "categorical" vs. "numerical" data, etc.)

---

# 1.2

## Feature types

- **Categorical** feature: non-numerical values&mdash;or "numerical" values w/o mathematical meaning.
    - e.g., "Yes"/"No", address, phone number (number but not mathematical), etc.
- **Numerical** feature: numerical values (with mathematical meaning).
    - e.g. amount spent on transaction, square footage of houes, etc.

## Plotting features in Python

`matplotlib` and `seaborn` are two popular libraries for plotting data. `seaborn` is based on the former and so can use functions & pallettes & stuff from `matplotlib`.

### Basic plot syntax

```py
sns.plot_type(data=df, x="x", y="y")
```

### Plotting fun

# 1.3: Modeling workflow in `scikit-learn`

## `scikit-learn` Modeling Workflow

1. Define **input and output features**.
   1. (pd dataframes)
2. **Initialize** an estimator.
   1. Import the estimator model (from a scikit-learn library) and instantiate it.
3. **Fit** the estimator.
   1. `.fit()`
   <!-- 1. `.fit()` to estimate model weights (or apply algorithm). -->
4. Use the fitted estimator to make **predictions**.
   1. `.predict()`
5. **Evaluate** the fitted estimator's performance.
   1. `.score()`

### Example

```py
from sklearn.linear_model import LinearRegression

# 1: Define input & output features
X = rides[["distance"]]
y = rides[["price"]]

# 2: Initialization
linearModel = LinearRegression() # instantiate an instance of the LinearRegression class

# 3: Fitting
linearModel = linearModel.fit(X, y)

# 4: Prediction
pred = linearModel.predict(X)

# 5: Evaluation
score = linearModel.score(X, y)
```

### Initializing Estimator - Syntax

To import & initialize a particular estimator w/ user-defined settings: 

```py
from sklearn.module import Estimator

estimatorObj = Estimator(settings go here) # leave blank for default settings (I presume?)
```

e.g.: 

```py
from sklearn.linear_model import LinearRegression

linearModel = LinearRegression(n_neighbors=7)
```

### Fitting Estimator (`.fit()`)

> `.fit(X, y)` estimates model params (e.g. regression weights), or applies an algorithm to in/output features.

`.fit(X, y)` **estimates model weights** (or applies algorithm) to input (`X`) & output features (`y`) (both pd datafrmaes), then **updates the model object** w/ the fitted estimator.

For models w/ coeffs (e.g. lin reg), the fitted estimator contains fitted weights in `coef_` and fitted intercept in `intercept_` (if applicable).  

```py
initializedModel = EstimatorModel() # initializing model
initializedModel.fit(input_features, output_values) # update model: estimate model weights (or apply algorithm to in/output features)

initializedModel.coef_ # coefficients
initalizedModel.intercept_ # intercept
```

### Making predictions (`.predict()`)

> `.predict(X)` calculates predicted vals for a set of inputs&mdash;either new vals or o.g. inputs.

Predicted values are denoted as $\hat y$, i.e. "y-hat".

`.predict(X)` returns predicted values. `X` can be original inputs or new data.

```py
pred = fittedModel.predict(X)
```

e.g.:

```py
X = pd.DataFrame({"distance": [5.0]})
pred = linearModel.predict(X)
```

### Evaluating Estimators

#### Metrics

- **Accuracy**: 
    - The proportion of instances correctly classified (in a classification model).
    - Ranges from $0.0$ to $1.0$. 
    - Higher indicates better estimator.
- $R^2$:
    - Approx. proportion of variation in the output feature "explained" by a regres model. 
        <!-- - i.e. "How much of the ups & downs we see in our target var can be attributed to the patterns the model learned from the input features?" -->
        - i.e. "What proportion of the variation in my output can I successfully predict using this model?"
    - Max value: $1.0$ (the estimator perfectly predicted the output feature).
    - Negative values mean the estimator predicted *worse* than predicting the **mean** output for all instances.
    - e.g. an $R^2$ of $0.102$ means that that linear model explains $10.2%$ of the variation for the input feature estimated.

#### `.score()`

> `.score(X, y)` calculates a performance metric based on the type of machine learning task.

You can call `.score()` (only) on supervised learning models. The return depends on the type of supervised learning model:

- **Classification** models will return **accuracy**
- **Regression** models will return $R^2$.

Calling `.score()` on an estimator using an **unsupervised** learning model will **throw an error**, because unsupervised learning tasks **do not have a known output** feature (so performance metrics cannot be calculated).

### Complete example (from Zybook)

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the tips dataset
dataTips = pd.read_csv("tips.csv")

# Create a dataframe X containing tip and total_bill
X = dataTips[["tip", "total_bill"]]

# Create a dataframe y containing day
y = dataTips[["day"]]

# Flatten y into an array
yNew = np.ravel(y)

# Initialize a GaussianNB() model
NBModel = GaussianNB()

# Fit the model to X and yNew
NBModel.fit(X, yNew)

# Determine the accuracy of the model NBModel
accuracy = NBModel.score(X, y)

# Print the accuracy
print(accuracy)
```

## Converting `pandas` dataframe to `numpy` array

`numpy.ravel(dataframe)` converts the `pandas` dataframe into a `numpy` array.