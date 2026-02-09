# 2.5: knn lab


## Instructions

The dataset SDSS contains 17 observational features and one class feature for 10000 deep sky objects observed by the Sloan Digital Sky Survey. Use scikit-learn's KNeighborsClassifier() function to perform kNN classification to classify each object by the object's redshift and u-g color. 

* Import the necessary modules for kNN classification
* Create dataframe X with features redshift and u_g
* Create dataframe y with feature class
* Initialize a kNN model with k = 3
* Fit the model using the training data
* Find the predicted classes for the test data
* Calculate the accuracy score using the test data
* Note that the code requires a user-defined random seed that must be entered

Ex: If a user-defined random seed of 42 is used, the output is:

    Accuracy score is 0.9837

## Code

```py
# Import needed packages for classification
# Import packages for evaluation
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Your code here

# Load the dataset
skySurvey = pd.read_csv("SDSS.csv")

# Create a new feature from u - g
skySurvey["u_g"] = skySurvey["u"] - skySurvey["g"]

# Create dataframe X with features "redshift" and "u_g"
X = skySurvey[["redshift","u_g"]]

# Create dataframe y with feature "class"
y = skySurvey[["class"]]

# Get user defined random seed
seed = int(input())
np.random.seed(seed)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Get user defined k
# k = int(input())
k = 3

# Initialize model with k=3
skySurveyKnn = KNeighborsClassifier(n_neighbors=k)

# Fit model using X_train and y_train
skySurveyKnn.fit(X_train, np.ravel(y_train))

# Find the predicted classes for X_test
y_pred = skySurveyKnn.predict(X_test)

# Calculate accuracy score
score = skySurveyKnn.score(X_test, y_test)

# Print accuracy score
print("Accuracy score is ", end="")
print("%.4f" % score)
```

# 2.6: Logistic Regression Lab

## Instructions

The `nbaallelo_log` file contains data on 126314 NBA games from 1947 to 2015. The dataset includes the features `pts`, `elo_i`, `win_equiv`, and `game_result`. Using the csv file `nbaallelo_log.csv` and `scikit-learn`'s `LogisticRegression()` function, construct a logistic regression model to classify whether a team will win or lose a game based on the team's `elo_i` score.

- Create a binary feature `win` for `game_result` with 0 for L and 1 for W
- Use the `LogisticRegression()` function with `penalty="l2"` to construct a logistic regression model with `win` as the target and `elo_i` as the predictor
- Print the weights and intercept of the fitted model
- Find the proportion of instances correctly classified

Note: Use `ravel()` from `numpy` to flatten the second argument of `LogisticRegression.fit()` into a 1-D array.

Ex: If the program uses the file `nbaallelo_small.csv`, which contains 100 instances, the output is:

    w1: [[3.64194406e-06]]
    w0: [-2.80257471e-09]
    0.5

*(that example was wrong for me, actually...so uh, idk)*

## Code

```py
# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Silence warning from sklearn
import warnings
warnings.simplefilter("ignore")

# Load nbaallelo_log.csv into a dataframe
NBA = pd.read_csv("nbaallelo_log.csv")

# Create binary feature for game_result with 0 for L and 1 for W
NBA["win"] = NBA["game_result"].map({"W": 1, "L": 0})
# alternatively: 
# NBA["win"] = np.where(NBA["game_result"] == "W", 1, 0)
# or even:
# NBA["win"] = (NBA["game_result"] == "W").astype(int)

# Store relevant columns as variables
X = NBA[["elo_i"]]
y = NBA[["win"]]

# Initialize and fit the logistic model using the LogisticRegression() function
model = LogisticRegression(penalty="l2")
model.fit(X, np.ravel(y))

# Print the weights for the fitted model
w1 = model.coef_
print("w1:", w1)

# Print the intercept of the fitted model
w0 = model.intercept_
print("w0:", w0)

# Find the proportion of instances correctly classified
score = model.score(X, y)
print(round(score, 3))
```

# 2.7: Naive Bayes Lab

## Instructions

The file `SDSS` contains 17 observational features and one class feature for 10000 deep sky objects observed by the Sloan Digital Sky Survey. Use `scikit-learn`'s `GaussianNB()` function to perform Gaussian naive Bayes classification on a subset of the data to classify each object by the object's redshift and u-g color.

- Import the necessary modules for Gaussian naive Bayes classification
- Create dataframe `X` with features `redshift` and `u_g`
- Create dataframe `y` with feature `class`
- Initialize a Gaussian naive Bayes model with the default parameters
- Fit the model
- Calculate the accuracy score
- Note that the code requires a user-defined random seed that must be entered


Note: Use `ravel()` from numpy to flatten the second argument of `GaussianNB.fit()` into a 1-D array.

Ex: If a user-defined random seed of 42 is used, the output is:

    Accuracy score is 0.992

## Code

```py
# Import the necessary modules
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the dataset
skySurvey = pd.read_csv("SDSS.csv")

# Get user defined random seed
seed = int(input())
np.random.seed(seed)
ssSample = skySurvey.sample(n=500)

# Create a new feature from u - g
ssSample["u_g"] = ssSample["u"] - ssSample["g"]

# Create dataframe X with features "redshift" and "u_g"
X = ssSample[["redshift","u_g"]]

# Create dataframe y with feature "class"
y = ssSample[["class"]]

# Initialize a Gaussian naive Bayes model
skySurveyNBModel = GaussianNB()

# Fit the model
skySurveyNBModel.fit(X, np.ravel(y))

# Calculate the proportion of instances correctly classified
score = skySurveyNBModel.score(X, y)

# Print accuracy score
print("Accuracy score is ", end="")
print('%.3f' % score)
```
