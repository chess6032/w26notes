# CS 270 Zybooks Ch. 4 code examples

- [4.1: CV methods](#41-cv-methods)
- [4.2: CV for model selection](#42-cv-for-model-selection)
- [4.3: Model tuning](#43-model-tuning)
- [4.4 LAB: Model selection using CV](#44-lab-model-selection-using-cv)
- [4.5 LAB: CV for selection hyperparams](#45-lab-cv-for-selection-hyperparams)

## 4.1: CV methods

### CV ex 1

```py
X = wine[["density", "alcohol"]]
y = wine[["type"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize k-nearest neighbors
knnModel = KNeighborsClassifier(n_neighbors=5)

# Fit k-nearest neighbors with 5-fold cross-validation
cv_results = cross_validate(knnModel, X, np.ravel(y), cv=5)

# View testing accuracy for each fold
print("Test score:", cv_results["test_score"])

# Plot accuracy for each fold
scores = cv_results["test_score"]

p = sns.swarmplot(x=scores)
p.set_xlabel("Cross-validation accuracy", fontsize=14)
p.set_ylabel("Count", fontsize=14)
```

### CV ex 2

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Split the `Dry_Bean_Data` dataset into stratified 70% training/validation and 30% testing sets with `random_state=151` and `stratify=y`.

The code provided contains all imports, loads the dataset, creates input and output feature sets, splits the training/validation dataset into training and validation datasets, prints the sizes of the split samples, and prints the test dataset.

```py
# Import packages and functions
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["roundness", "Eccentricity"]]
y = beans[["Class"]]

# Set aside 30% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=151, stratify=y)

# Split training again into 60% training and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1/(1-0.3), random_state=151, stratify=y_train)

# Print split sizes and test dataset
print("original dataset:", len(beans), 
    "\ntrain_data:", len(X_train), 
    "\nvalidation_data:", len(X_val), 
    "\ntest_data:", len(X_test),
    "\n", X_test
)
```

### CV ex 3

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Split the `Dry_Bean_Data` dataset into stratified 60% training/validation and 40% testing sets with `random_state=89` and `stratify=y`.
- Split the training/validation set into training and validation sets with `random_state=89` so that the final split is 50% training, 10% validation, and 40% testing.

The code provided contains all imports, loads the dataset, creates input and output feature sets, prints the sizes of the split samples, and prints the test dataset.

```py
# Import packages and functions
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["Solidity", "EquivDiameter"]]
y = beans[["Class"]]

# Set aside 40% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=89, stratify=y)

# Split training again into 50% training and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1/(1-0.4), random_state=89, stratify=y_train)

# Print split sizes and test dataset
print("original dataset:", len(beans), 
    "\ntrain_data:", len(X_train), 
    "\nvalidation_data:", len(X_val), 
    "\ntest_data:", len(X_test),
    "\n", X_test
)
```

### CV ex 4

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Fit a linear discriminant analysis model with 5-fold cross-validation.
- Print the test score.

The code provided imports the dataset and packages, creates input and output feature sets, initializes a linear discriminant analysis model, `beansModel`, and prints the test score.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["ConvexArea", "Extent"]]
y = beans[["Class"]]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
beansModel = LinearDiscriminantAnalysis(n_components=2, store_covariance=True)

# Fit linear discriminant analysis model with 5-fold cross-validation
beanResults = cross_validate(beansModel, X, y, cv=5)

# Print test score
print("Test score:", beanResults["test_score"])
```

## 4.2: CV for model selection

### `cross_validate()` and `KFold()` 

```py
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

# read data
wine = pd.read_csv("wine_sample.csv")
X = wine[["sulphates", "alcohol"]]
y = wine[["type"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123
)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define candidate models
knnModel = KNeighborsClassifier(n_neighbors=5)
logisticModel = LogisticRegression()

# Define a set of cross-validation folds
kf = KFold(n_splits=10, random_state=19, shuffle=True)

# Fit k-nearest neighbors with 10-fold cross-validation to the training data
knnResults = cross_validate(knnModel, X_train, np.ravel(y_train), cv=kf)

knnScores = knnResults["test_score"]

# View accuracy for each fold
print("k-nearest neighbor scores:", knnScores.round(3))

# Calculate descriptive statistics
print("Mean:", knnScores.mean().round(3))
print("SD:", knnScores.std().round(3))

# Logistic model performance on testing
knnModel.fit(X_train, np.ravel(y_train))
knnModel.score(X_test, np.ravel(y_test))

# Fit logistic regression with 10-fold cross-validation to the training data
logisticResults = cross_validate(logisticModel, X_train, np.ravel(y_train), cv=kf)

logisticScores = logisticResults["test_score"]

# View accuracy for each fold
print("Logistic regression scores:", logisticScores.round(3))

# Calculate descriptive statistics
print("Mean:", logisticScores.mean().round(3))
print("SD:", logisticScores.std().round(3))

# Logistic model performance on testing
logisticModel.fit(X_train, np.ravel(y_train))
logisticModel.score(X_test, np.ravel(y_test))

# Combine scores from both models into a dataframe
df = pd.DataFrame({"knn": knnScores, "logistic": logisticScores})

# Boxplot of errors for k-nearest neighbors
p = sns.boxplot(data=df, y="knn")
p.set_xlabel("k-nearest neighbors", fontsize=14)
p.set_ylabel("Cross-validation scores", fontsize=14)

# Boxplot of errors for logistic regression
p = sns.boxplot(data=df, y="logistic")
p.set_xlabel("Logistic regression", fontsize=14)
p.set_ylabel("Cross-validation scores", fontsize=14)
```

### 4.2 Ex 1

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Define a set of 10 cross-validation folds with `random_state=35` and `shuffle=True`.

The code provided contains all imports, loads the dataset, creates input and output feature sets, initializes a linear discriminant analysis model and a Gaussian naive Bayes model, fits the models with 10-fold cross validation, and prints the descriptive statistics for each model.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["Extent", "Eccentricity"]]
y = beans[["Class"]]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a set of 10 cross-validation folds
kf = KFold(n_splits=10, random_state=35, shuffle=True)

# Initialize the linear discriminant analysis model with two components
LDAmodel = LinearDiscriminantAnalysis(n_components=2)

# Fit linear discriminant analysis model with cross-validation
LDAresults = cross_validate(LDAmodel, X, np.ravel(y), cv=kf)

beansLDAScores = LDAresults["test_score"]

# View accuracy for each fold
print("Linear discriminant analysis scores:", beansLDAScores.round(3))

# Calculate descriptive statistics
print("Mean:", beansLDAScores.mean().round(3))
print("SD:", beansLDAScores.std().round(3))

# Initialize the Gaussian naive Bayes model
NBmodel = GaussianNB()

# Fit Gaussian naive Bayes model with 10-fold cross-validation
NBBeanResults = cross_validate(NBmodel, X, np.ravel(y), cv=kf)

modelNBScores = NBBeanResults["test_score"]

# View accuracy for each fold
print("Gaussian naive Bayes scores:", modelNBScores.round(3))

# Calculate descriptive statistics
print("Mean:", modelNBScores.mean().round(3))
print("SD:", modelNBScores.std().round(3))
```

### 4.2 Ex 2

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Initialize a linear discriminant analysis model `beansLDAModel` with two components.
- Fit the model with cross validation, using `kf` as the number of folds.

The code provided contains all imports, loads the dataset, creates input and output feature sets, defines a set of 9 cross-validation folds, initializes a Gaussian naive Bayes model, fits the model with 9-fold cross validation, and prints the descriptive statistics for each model.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["Eccentricity", "Solidity"]]
y = beans[["Class"]]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a set of 9 cross-validation folds
kf = KFold(n_splits=9, random_state=25, shuffle=True)

# Initialize the linear discriminant analysis model with two components
beansLDAModel = LinearDiscriminantAnalysis(n_components=2)

# Fit linear discriminant analysis model with cross-validation
LDABeanResults = cross_validate(beansLDAModel, X, y, cv=kf)

beansLDAScores = LDABeanResults["test_score"]

# View accuracy for each fold
print("Linear discriminant analysis scores:", beansLDAScores.round(3))

# Calculate descriptive statistics
print("Mean:", beansLDAScores.mean().round(3))
print("SD:", beansLDAScores.std().round(3))

# Initialize the Gaussian naive Bayes model
beansNB = GaussianNB()

# Fit Gaussian naive Bayes model with cross-validation
modelNBResults = cross_validate(beansNB, X, np.ravel(y), cv=kf)

NBBeanScores = modelNBResults["test_score"]

# View accuracy for each fold
print("Gaussian naive Bayes scores:", NBBeanScores.round(3))

# Calculate descriptive statistics
print("Mean:", NBBeanScores.mean().round(3))
print("SD:", NBBeanScores.std().round(3))
```

### 4.2 Ex 3

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Initialize a Gaussian naive Bayes model `NBmodel`.
- Fit the model with cross validation, using `kf` as the number of folds.

The code provided contains all imports, loads the dataset, creates input and output feature sets, defines a set of 5 cross-validation folds, initializes a linear discriminant analysis model, fits the model with 5-fold cross validation, and prints the descriptive statistics for each model.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["Eccentricity", "Compactness"]]
y = beans[["Class"]]

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a set of 5 cross-validation folds
kf = KFold(n_splits=5, random_state=12, shuffle=True)

# Initialize the linear discriminant analysis model with two components
beansLDA = LinearDiscriminantAnalysis(n_components=2)

# Fit linear discriminant analysis model with cross-validation
modelLDAResults = cross_validate(beansLDA, X, np.ravel(y), cv=kf)

modelLDAScores = modelLDAResults["test_score"]

# View accuracy for each fold
print("Linear discriminant analysis scores:", modelLDAScores.round(3))

# Calculate descriptive statistics
print("Mean:", modelLDAScores.mean().round(3))
print("SD:", modelLDAScores.std().round(3))

# Initialize the Gaussian naive Bayes model
NBmodel = GaussianNB()

# Fit Gaussian naive Bayes model with cross-validation
NBBeanResults = cross_validate(NBmodel, X, y, cv=kf)

beansNBScores = NBBeanResults["test_score"]

# View accuracy for each fold
print("Gaussian naive Bayes scores:", beansNBScores.round(3))

# Calculate descriptive statistics
print("Mean:", beansNBScores.mean().round(3))
print("SD:", beansNBScores.std().round(3))
```

## 4.3: Model tuning

### `GridSearchCV()`

```py
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

# read data
wine = pd.read_csv("wine_sample.csv")

X = wine[["sulphates", "alcohol"]]
y = wine[["type"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123
)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize k-nearest neighbors model
knnModel = KNeighborsClassifier()

# Create tuning grid
k = {"n_neighbors": [3, 5, 7, 9, 11]}

# Initialize tuning grid and fit to training data
knnTuning = GridSearchCV(knnModel, k)
knnTuning.fit(X_train, np.ravel(y_train))

# All available results are stored here:
knnTuning.cv_results_

# Mean testing score for each k and best model
print("Mean testing scores:", knnTuning.cv_results_["mean_test_score"]) # mean validation scores for each hyperparam across CV iterations
print("Best estimator:", knnTuning.best_estimator_)
```

### `validation_curve()`

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

wine = pd.read_csv("wine_sample.csv")
X = wine[["sulphates", "alcohol"]]
y = wine[["type"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=123
)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply 5-fold cross-validation over tuning grid using validation_curve
k = [3, 5, 7, 9, 11]
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(),
    X_train,
    np.ravel(y_train),
    param_range=k,
    param_name="n_neighbors",
    cv=5,
)

# Calculate mean and SD for training and testing
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot cross-validation results on training/validation for each parameter value
sns.lineplot(x=k, y=train_scores_mean, label="Training", color="#1f77b4")
sns.lineplot(x=k, y=test_scores_mean, label="Validation", color="#ff7f0e")
plt.fill_between(
    k,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="#1f77b4",
)
plt.fill_between(
    k,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="#ff7f0e",
)

plt.xlabel("k", fontsize=16)
plt.ylabel("Score", fontsize=16)
```

### 4.3 Ex. 1

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Create a tuning grid for cross validation tuning for k-nearest neighbors classification parameters. Use the values $k$ = 2, 9, 10, 11.

The code provided contains all imports, loads the dataset, creates input and output feature sets, splits the data into training and testing sets, initializes a k-nearest neighbor model, initializes and fits a tuning grid to the training data, and prints the results.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["EquivDiameter", "roundness", "Eccentricity"]]
y = beans[["Class"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize k-nearest neighbors model
knnModel = KNeighborsClassifier()

# --------------------------------------------
# Create tuning grid
k = {"n_neighbors": [2, 9, 10, 11]}
# --------------------------------------------

# Initialize tuning grid and fit to training data
knnTuning = GridSearchCV(knnModel, k)
knnTuning.fit(X_train, np.ravel(y_train))

# Print mean testing scores
print("Mean testing scores:", knnTuning.cv_results_["mean_test_score"])
```

### 4.3 Ex. 2

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Initialize a tuning grid `beanGrid` for 6-fold cross validation for a k-nearest neighbors model, with the parameter `param_grid` set to `k`.
- Fit the tuning grid to the training data.

The code provided contains all imports, loads the dataset, creates input and output feature sets, splits the data into training and testing sets, initializes a k-nearest neighbor model, creates a tuning grid, and prints the mean test score.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["Solidity", "roundness", "Extent"]]
y = beans[["Class"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize k-nearest neighbors model
kNNBean = KNeighborsClassifier()

# Create tuning grid
k = {"n_neighbors": [6, 8, 10, 11]}

# --------------------------------------------
# Initialize tuning grid 
beanGrid = GridSearchCV(kNNBean, k, cv=6)

# Fit grid to training data
beanGrid.fit(X_train, np.ravel(y_train))
# --------------------------------------------

# Print mean testing scores
print("Mean testing scores:", beanGrid.cv_results_["mean_test_score"])
```

### 4.3 Ex. 3

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Use the attribute `best_params_` to print the value of `k` that results in the best mean testing score.

The code provided contains all imports, loads the dataset, creates input and output feature sets, splits the data into training and testing sets, initializes a k-nearest neighbor model, creates a tuning grid, and initializes and fits a tuning grid to the training data.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the dry beans dataset
beans = pd.read_csv("Dry_Bean_Data.csv")

# Create input and output feature sets
X = beans[["ConvexArea", "roundness", "EquivDiameter"]]
y = beans[["Class"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize k-nearest neighbors model
kNNModel = KNeighborsClassifier()

# Create tuning grid
k = {"n_neighbors": [3, 8, 10, 11]}

# Initialize tuning grid 
beanGrid = GridSearchCV(kNNModel, k, cv=5)

# Fit grid to training data
beanGrid.fit(X_train, np.ravel(y_train))

# --------------------------------------------
# Print value of k that results in the best mean testing score
print("Best parameter:", beanGrid.best_params_)
# --------------------------------------------
```


## 4.4 LAB: Model selection using CV

The `taxis` dataset contains information on taxi journeys during March 2019 in New York City. The data includes time, number of passengers, distance, taxi color, payment method, and trip locations. Use `scikit-learn`'s `cross_validate()` function to fit a linear regression model and a k-nearest neighbors regression model with 10-fold cross-validation.

- Create dataframe `X` with the feature distance.
- Create dataframe `y` with the feature fare.
- Split the data into 80% training, 10% validation and 10% testing sets, with `random_state=42`.
- Initialize a linear regression model.
- Initialize a k-nearest neighbors regression model with k = 3.
- Define a set of 10 cross-validation folds with random_state=42.
- Fit the models with cross-validation to the training data, using the default performance metric.
- For each model, print the test score for each fold, as well as the mean and standard deviation for the model.


```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

taxis = pd.read_csv("taxis.csv")

# Create dataframe X with the feature distance
X = taxis[["distance"]]
# Create dataframe y with the feature fare
y = taxis[["fare"]]

# Set aside 10% of instances for testing
TEST_SIZE = 0.1
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# Split training again into 80% training and 10% validation 
VALIDATION_SIZE = 0.1
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE / (1 - TEST_SIZE), random_state=42)

# Initialize a linear regression model
SLRModel = LinearRegression()
# Initialize a k-nearest neighbors regression model with k = 3
knnModel = KNeighborsRegressor(n_neighbors=3)

# Define a set of 10 cross-validation folds with random_state=42
kf = KFold(n_splits=10, random_state=42, shuffle=True)

# Fit k-nearest neighbors with cross-validation to the training data
knnResults = cross_validate(knnModel, X_train, np.ravel(y_train), cv=kf)

# Find the test score for each fold
knnScores = knnResults["test_score"]
print("k-nearest neighbor scores:", knnScores.round(3))

# Calculate descriptive statistics for k-nearest neighbor model
print("Mean:", knnScores.mean().round(3))
print("SD:", knnScores.std().round(3))

# Fit simple linear regression with cross-validation to the training data
SLRModelResults = cross_validate(SLRModel, X_train, np.ravel(y_train), cv=kf)

# Find the test score for each fold
SLRScores = SLRModelResults["test_score"]
print("Simple linear regression scores:", SLRScores.round(3))

# Calculate descriptive statistics simple linear regression model
print("Mean:", SLRScores.mean().round(3))
print("SD:", SLRScores.std().round(3))
```

## 4.5 LAB: CV for selection hyperparams

The diamond dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds. This data can be used to predict the price of a diamond based on its characteristics. Use `scikit-learn`'s `GridSearchCV()` function to train and evaluate an elastic net model over a hyperparameter grid.

- Create dataframe `X` with the features `carat` and `depth`.
- Create dataframe `y` with the feature `price`.
- Split the data into 80% training and 20% testing sets, with `random_state = 42`.
- Initialize an elastic net model with `random_state = 0`.
- Create a tuning grid with the hyperparameter name `alpha` and the values `0.1`, `0.5`, `0.9`, `1.0`.
- Use `GridSearchCV()` with `cv=10` to initialize and fit a tuning grid to the training data.
- Print the mean testing score for each fold and the best parameter value.

```py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

diamonds = pd.read_csv("diamonds.csv")

# Create dataframe X with the features carat and depth
X = diamonds[["carat", "depth"]]
# Create dataframe y with the feature price
y = diamonds[["price"]]

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize elastic net model
ENModel = ElasticNet(random_state=0)

# Create tuning grid
alpha = {"alpha": [0.1, 0.5, 0.9, 1.0]}

# Initialize tuning grid and fit to training data
ENTuning = GridSearchCV(ENModel, alpha, cv=10)
ENTuning.fit(X_train, np.ravel(y_train))

# Mean testing score for each lambda and best model
print("Mean testing scores:", ENTuning.cv_results_["mean_test_score"])
print("Best estimator:", ENTuning.best_estimator_)
```
