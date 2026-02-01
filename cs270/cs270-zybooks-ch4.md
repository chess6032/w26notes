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



# 4.3


# 4.4 (LAB)


# 4.5 (LAB)

