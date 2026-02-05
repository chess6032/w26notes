ZYBOOKS CHAPTER 4: MODEL VALIDATION

---

# 4.1: Cross-validation Methods

> Learn to:
> 
> - Explain why cross-validation is important in machine learning.
> - Define training, validation, and testing sets.
> - Explain when to use stratified splits.
> - Create training, validation, and testing sets in scikit-learn.
> - Define k-fold cross-validation.
> - Define leave-one-out cross-validation.
> - Implement cross-validation in scikit-learn.

## Cross-validation (CV)

- **Model training** is the process of ESTIMATING PARAMS for an ML model.
  - During mod training, params are estimated to MINIMIZE PREDICTION ERRS or MAXIMIZE PREDICTION ACCURACY on the output feature.
- **Model validation** is the process of EVALUATING a model's initial performance and ADJUSTING param estimates or hyperparam settings if needed.
- **Model testing** is the process of evaluating a model's FINAL performance.

Ideally, an unseen set of data would be available to test the trained model. But gathering more data isn't always practical (can be expensive and time-consuming), so other model evaluation techniques are needed.

- In **CROSS-VALIDATION (CV)**, instead of gathering training & testing data separately, you use one dataset and split it into SUBSETS for model training & model testing.
  - Reserving a subset of data for testing (and not for training) allows fitted models to be evaluated w/o risk of bias that occurs when you test a model on the same data it trained on.

CV can be used to fine-tun a model's hyperparams, or to choose btwn competing models.

## Data splits

Dividing your dataset into subsets for CV is conventionally called "splitting" your data. You oft split it into three subsets: 

- **TRAINING SET**. Used to&mdash; 
  - Estimate model PARAMS.
  - FIT the initial model.
- **VALIDATION SET**. Used to&mdash;
  - Evaluate the model's INITIAL PERFORMANCE.
  - Decide OPTIMAL HYPERPARM VALS
  - Assess whether a model is OVER/UNDER FITTED.
- **TESTING SET**. Used to&mdash; 
  - Evaluate a model's FINAL PERFORMANCE.
  - SELECT btwn COMPETING MODELS.

<!-- i.e.:

| Subset     | Purpose                                   |  
| ---------- | ----------------------------------------- |  
| Training   | Estimate model params.                    |  
| Validation | Evaluate the model's initial performance. |  
| Testing    | Evaluate the model's final performance.   |   -->

<br/>

- Sometimes, it's only necessary to split your data into training & testing sets, like if your model has no hyperparams (e.g. simple linear regression).
- Most of the data should be assigned to training, but the exact split you choose depends on how much data you have and what your goal of CV is.
  - A higher ratio of training data means the data used to fit the model is a better representation of the dataset. BUT on the flipside, more training data makes it more expensive to train the model. 
- Using RANDOM SAMPLING to create your splits avoids bias, since it means all instances are equally likely to be in any of the splits.

### Stratified splits

<!-- Typically, data is split into its subsets randomly. However, this can cause some features to not be evenly distributed btwn your subsets. This is a problem when you have a particular class that is very important but rare. You'd want that class to be evenly represented in all the subsets of your split.  -->

- **STRATIFIED CV SETS** are EVENLY SPLIT for ALL LVLS OF THE OUTPUT FEATURE. It does this by performing RANDOM SPLITS WITHIN EACH CLASS, as opposed to randomly splitting the dataset as a whole.
  - i.e., EVERY FEATURE IS EVENLY REPRESENTED IN EACH SPLIT.
  - In classification, this means that THE CLASS PROPORTIONS IN EACH CV SET reflects the class props for the whole dataset.
  - ^ Hence, stratified splits helps avoid **UNDERFITTING**.
  - Stratification is important when:
    - You have class/value that is important but rare.
    - The fitted model depends on class proportions.

### Splitting data in scikit-learn

- `training_test_split()` from `sklearn.model_selection`
  - Required params: `X` (input features) and `y` (output features).
  - Optional params (default for all is `None`):
    - `test_size`: float btwn `0.0` and `1.0`. Sets the props of instances to allocate to testing.
    - `train_size`: float btwn `0.0` and `1.0`. Sets the props of instances to allocate to training.
    - `random_state`: sets seed for randomization. Using `random_state` ensures the random allocation can be reproduced later.
    - `stratify`: assign an output feature to `stratify` to create random splits within each class.
  - Return:
    - Tuple: `X_train`, `X_test`, `y_train`, `y_test`.

#### Using `stratify`

```py
X = data[["input1", "input2"]]
y = data[["output"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

#### Making a validation subset

`training_test_split()` only returns a split into training data and test data. To make a validation subset, split the dataset into training & testing, then split one of those to create the validation subset.

Here's an example where we split the data into "training"/testing, and then the "training" into training/validation:

```py
X_temp, X_test, y_temp, y_test = training_test_split(X, y, test_size=TEST_SIZE)
X_train, X_validation, y_train, y_validation = training_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE / (1 - TEST_SIZE))
```

## $k$-fold CV

A single round of CV is oft not enough to evaluate a model. Its params and performance may be highly variable depending on what's in each CV subset. $k$-fold CV is a way of doing multiple iterations of CV, each time with different splits, to avoid this. 

- **$k$-FOLD CV** splits a training set into $k$ NON-OVERLAPPING SUBSETS, each called "FOLDS". Then, you do CV $k$ times: Each time, ONE FOLD is used for VALIDATION, while the others are used for training.
  - Hence, in each iteration there is $1$ fold used for validation and $k-1$ folds used for training.
  - USAGE: 
    - Since $k$-fold CV performs CV multiple times, it may be used to MEASURE THE VARIABILITY OF PARAMS & PERFORMANCE MEASURES.
    - $k$-fold CV may be used to SELECT POSSIBLE HYPERPARAM VALS, or MEASURE HOW SENSITIVE a model's performance is to CV SPLITS.
  - To clarify: You make the folds at the beginning&mdash;you don't change what's in each fold. But with each iteration, you choose a different fold to be the validation subset.
  - The MEAN CV SCORE represents the AVG CLASSIFICATION ACCURACY across all folds.
  - Using more folds is computationally expensive, so 5-10 FOLDS IS RECOMMENDED.

## Leave-one-out CV (LOOCV)

LOOCV can be thought of as $k$-fold CV w/ a single instance in each fold.

- **Leave-one-out CV (LOOCV)** holds out ONE INSTANCE AT A TIME for validation, w/ the remaining instances used to train the model.
  - USAGE:
    - Useful for identifying INDIVIDUAL INSTANCES W/ A STRONG INFLUENCE ON A MODEL.
      - e.g., when using a multiple regression model, which is sensitive to outliers.
  - In each iteration of LOOCV, you calculate the MSE. Instances that resulted in an abnormally high MSE when left out are considered outliers.
    - Why MSE? Remember that MSE = $\frac 1 n \sum^n_{i=1}(\hat y_i - y_i)^2$. When $n=1$, this reduces to $(\hat y_i - y_i)^2$. So in LOOCV, "MSE equals the squared residual" (whatever tf that means).
  - For a dataset w/ $n$ instances, a complete LOOCV would take $n$ iterations. Hence, it is quite computationally expensive.

## CV in Python

- `cross_validate(estimator, X, y)` from `sklearn.model_selection`.
  - Implements a specified model (`estimator`) to input/output features (`X`/`y`) using **$k$-fold CV**.
  - (Optional) Params:
    - `estimator`: An sklearn model object or function. (Default: `None`...idk wtf it would do in that case.)
    - `scoring`: list, specifying one or more performance metrics. (Default: `None`.)
      - For CLASSIFICATION models: default performance metric is ACCURACY.
      - For REGRESSION models, the default is $R^2$.
      - One option is `"neg_mean_squared_error`, which gives you the negative MSE. You can multiply this by -1 to obtain the actual MSE.
      - For a list of available performance metrics, see the [scikit-learn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
    - `cv`: int. The number of folds (for $k$-fold CV). (Default: `5`.)
      - OR: A set of folds (made with `KFold()`.)
    - `return_train_score`: bool. Specifies whether/not to include performance metrics for each training set (huh???). (Default: `False`.)
    - `return_estimator`: bool. Specifies whether/not to return the estimated model params for each training set (huh???). (Default: `False`.)
    - `params`: dictionary. Specifies hyperparams for the CV model. (Default: `None`.) 
  - Return:
    - Dictionary. Model performance metrics on the testing folds are stored in an array linked to the key `test_score`&mdash;and other stuff, if requested (see params).
  - For more info, see [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate).

### Examples

#### CV ex 1

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

#### CV ex 2

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

#### CV ex 3

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

#### CV ex 4

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

# 4.2

## Model selection

**Model selection** is the process of identifying the best of a bunch of fitted models.

Criteria for good models:

- **STRONG PERFORMANCE**. (e.g.: high accuracy, low MSE.)
- **CONSISTENT PERFORMANCE**. (e.g.: models perform similarly during CV, or across multiple train/val/test splits.)
  - e.g. a model w/ validation scores ranging from 0.7-0.8 is better tahn a model w/ scores ranging 0.5-0.9.
- **REASONABLE ASSUMPTIONS**. (e.g.: No model assumptions are seriously violated. (Idk what this means.))

## Model selection w/ CV

CV is useful for model selection bc each potential model is trained across multiple splits (so you're seeing how each performs w/ different data). A good model should have strong performance metrics w/ low variability across ALL CV splits.

STEPS:

1. Define a set of CV splits.
2. Train and validate all potential models on every split.
3. Compare performance metrics on the validation sets ("using descriptive statsitics or summary plots").

The same CV splits shouold be used on each model to avoid bias. I.e., performance metrics on different data can't be compared.

### Comparing performance metrics

Look at models' average performances across all folds.

- When a few folds have relatively high or low scores, the MEDIAN should be used to compare models instead of the mean.

### CV params

Param estimates&mdash;e.g. weight in log reg&mdash;depend on train/val/test splits. Models w/ high param variability or skewness may suggest a poor fit to the dataset.

- If a model's param estimates are HIGHLY VARIABLE, then the model may be OVERFITTING to each training set.

## CV Model selection in sklearn

### `cross_validate()`

sklearn has `cross_val_score()` and `cross_validate()`, but the former only returns scores while the latter returns a lot more information in addition to scores, so `cross_validate()` is preffered.

#### `cross_validate()` returns

`cross_validate()` returns a DICTIONARY. Associated w/ each key is an array where each element corresponds to the result of one CV split. The table below includes the dictionaries keys and a description of what is contained in each element of the associated array.

| Key             | Contents of associated array |  
| --------------- | ---------------------------- |  
| `"test_score" ` | Validation scores. |  
| `"train_score"` | Training scores. Only returned if `return_train_score=True` is passed into `cross_validate()`. |  
| `"fit_time"`    | Time for fitting the estimator. |  
| `"score_time"`  | Time for scoring the estimator. |  
| `"estimator"`   | Fitted models. Only returned if `return_estimator=True` is passed into `cross_validate()`. |  

### `KFold()`

For model comparisons to be valid, **the same set of CV folds must be used across models**. 

To do this in code, use `KFold()` to define a set of CV folds. Then pass that in as the `cv` param each time you use `cross_validate()` with your models.

#### `KFold()` params

| Param          | Default | Description |  
| -------------- | ------- | ----------- |  
| `n_splits`     | 5       | INTEGER. Number of folds. Must be at least 2. |  
| `shuffle`      | False   | BOOL. Whether/not to shuffle the data before splitting. (Note that samples within each split will not be shuffled.) |  
| `random_state` | `None`    | INTEGER. Sort of like setting a seed for the shuffling. (`shuffle` MUST be set to TRUE when you pass in a val for `random_state`, otherwise you'll get an error at runtime.) |  

For more info, see sklearn's [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) on `KFold()`.

### Example

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

### Final model selection

Model selection identifies the best model type&mdash;but not necessarily the best params of the best model. (e.g. KNN does not have params that must be trained, but log reg does.) 

- Models WITHOUT trained params can be EVALUATED DIRECTLY on the testing set. 
- Models WITH trained params MUST BE FITTED to the entire training set BEFORE EVALUATING on testing.
  - Note that while you trained on only part of the training set during CV model selection, when you select your model you fit it to the ENTIRE TRAINING SET before testing.
    - (This avoids selection bias.)
  - Do not select the best params from a single CV fold. That introduces (selection?) bias.

### Examples

#### 4.2 Ex 1

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

#### 4.2 Ex 2

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

#### 4.2 Ex 3

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

# 4.3

## Model tuning

- **HYPERPARAMETER**: *User-defined* setting in a ML NOT estimated during model fitting. 
  - (e.g. $k$ in KNN or $\alpha$ in ridge regression.)
- **MODEL TUNING**: the process of SELECTING THE BEST HYPERPARAMS for a model USING CV.

Not all ML models take hyperparams, so therefore not all models require model tuning.

### Tuning grids

- **TUNING GRID**: LIST OF HYPERPARAM VALS to evaluate during model tuning.
  - Typically ~2-10 vals for each hyperparam.
  - Vals may be equally spaced, randomly selected, or chosen based on context.

When MORE THAN ONE HYPERPARAM exists, ALL *COMBINATIONS* ARE TESTED. Therefore, tuning grids w/ too many vals or too many hyperparams will be computationally intensive.

- **MULTISTAGE MODEL TUNING**: the process of using MULTIPLE STAGES to find the optimal hyperparam: At the end of each stage, you REFINE THE TUNING GRID and USE A NEW SET OF CV FOLDS TO EVALUATE the refined grid.
  - If a precise hyperparam is required, multistage tuning should be used.

### Selecting optimal hyperparam

An "optimal hyperparam" is a hyperparam that performs well in CV. Often, no single optimal hyperparam exists, and several hyperparams may be acceptable.

## Model tuning w/ tuning grids in sklearn

`GridSearchCV()` trains & evaluates models over a hyperparam grid. ([Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html))

If specific hyperparam values are unknown, `RandomizedSearchCV(estimator, param_distributions)` samples hyperparams from a given probability distribution (`param_distributions`). ([Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV))

### `GridSearchCV()` params

| Param        | Default | Description                                                                                                                                           |  
| ------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |  
| `estimator`  | NA      | SKLEARN MODEL. Can be passed as an initialized model or the model function. (e.g. `estimator=KNeighborsClassifier()`.)                                |  
| `param_grid` | NA      | DICTIONARY. Each KEY is a **hyperparam name** (STRING); each key's VALUE is an ARRAY of **hyperparam values** to try.                                 |  
| `cv`         | 5       | INTEGER for the **number of CV folds**, or a set of predefined CV folds.                                                                              |  
| `verbose`    | 0       | INTEGER specifying **how much info to print to the console** while fitting models. Possible vals range from 0-4, w/ higher vals printing more detail. |  
| `refit`      | `True`  | BOOL. Setting `refit=True` **re-trains the model** using the best hyperparams.                                                                        |  

### `GridSearchCV()` results

After completing the grid search:

- Results are stored as a DICTIONARY in the attribute `.cv_results_`.
- The best estimator is contained in the attribute `.best_estimator_`.
- After using `GridSearchCV()` on a model, calling that model's `.predict()` and `.score()` will only return results for the BEST hyperparams.

### Example

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

## Validation Curves

- **VALIDATION CURVE**: Plot the mean CV scores for each hyperparam val that was tested in model tuning.
  - Useful for detecting OVERFITTING: Overfitted hyperparams will have HIGH TRAINING SCORES & LOW VALIDATION SCORES.
    - THE BIGGER THE GAP BTWN TRAINING & VALIDATION SCORES, THE MORE OVERFITTED THE MODEL IS.
  - Optimal hyperparams will perform well on training AND validation sets.

## Validation curves in sklearn

`validation_curve()` implements grid search CV for a sklearn model on a single hyperparam.
It returns TRAINING & VALIDATION SCORES in a FORMAT THAT'S NICE FOR PLOTTING.

### `validation_curve()` params

| Param         | Default | Description |  
| ------------- | ------- | ----------- |  
| `estimator`   | NA      | MODEL FUNCTION. Sets the model to use for CV. (e.g. `estimator=KNeighborsClassifier()`) |  
| `param_name`  | NA      | STRING. Sets the param name for the specified function. (e.g. `param_name="n_neighbors"`) |  
| `param_range` | NA      | ARRAY. Contains param values to test. |  
| `cv`          | 5       | INTEGER for the number of CV folds, or a set of predefined CV folds. |  
| `verbose`     | 0       | INTEGER specifying how much info to print to console during model fitting. May be btwn 0-4, w/ higher vals printing more detail. |  

### `validation_curve()` returns

`validation_curve()` returns a TUPLE OF ARRAYS: the first item in the tuple contains the training scores, and the second contains the test scores.

### Example

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

## Examples

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


# 4.4 (LAB)

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

# 4.5 (LAB)

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
