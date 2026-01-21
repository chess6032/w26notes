# 2.1: $k$-Nearest Neighbors

## TODO:

* knn algorithm  
* "hyperparameter"  
* choosing appropriate $k$
* "decision boundaries"
    * effects of $k$ & sample size on decision boundaries
        * (maximum suggested $k$ depends on sample size)
* "distance measures"
* euclidean distance vs. manhattan distance vs. minkowski distance.
* knn in `scikit-learn`
    * which functions/classes/libraries to use
        * e.g. `KNeighborsClassifier`
    * parameters for those functions (& their defaults)
        * e.g. `n_neighbors` (and I think also `metric` and one other important one they mentioned)
    * `StandardScaler` (from `sklearn.preprocessing`) and `train_test_split()` (from `sklearn.model_selection`)
* "distance-based algorithms"
* "standardized features" (vs. "unstandardized features")


# 2.2: Logistic Regression

## TODO:

* "model-based classification"
* linear regression
    * formula
* logistic regression
    * formulas
    * use for classification
        * thresholds
    * logistic regression w/ multiple features
        * formulas
    * logistic regression in `scikit-learn`
        * functions/classes/libraries
        * `LogisticRegression()` parameters (& their defaults)
            * `penalty`
            * `max_iter`
        * function for predicting probabilities (`model.predict_proba(X)`) (as opposed to predicting classifications (`model.predict(X)`))
    * "cutoff"
* "binary classification" (maybe this has been defined in a previous chapter or section)

# 2.3: Gaussian Naive Bayes

* Baye's rule
    * formulas (there's so much probs/stats!!!)

$$P(A|B) = \frac{P(B|A)\times P(A)}{P(B)}$$

* "conditional probability"
* naive bayes classifier
    * formula
* "prior probability"
* "posterior probability"

$$P(y_i|x) = \frac{P(x|y_i)\times P(y_i)}{P(x)}$$

* assumptions of naive Bayes classifiers:
    * assumptions (for inputs):
        * independent/uncorrelated
        * equally important
    * (always met for single-input feature)
* gaussian naive Bayes
* "continuous probability distribution"
* "normal distribution" ($\text{normal}(\mu,\sigma)$)
    * $\mu$ ("mean")
    * $\sigma$ ("standard deviation") (i.e. 'spread')

$$f(x)=\frac{1}{2\pi\sigma^2}\exp(-(x-\mu)^2/2\sigma^2)$$

* "bayesian models"
* "bayesian priors"
* "uniform prior"
* naive Bayes in `scikit-learn`
    * `GaussianNB()`
        * `priors` parameter
* advantages & disadvantages of naive Bayes
* timing cell runtimes in Jupyter notebooks (`%%time`)

# 2.5: knn Lab

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