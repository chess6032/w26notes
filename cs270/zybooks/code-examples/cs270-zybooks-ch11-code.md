# CS 270 Ch. 11 Code Examples

## 11.1.1: Filter-based feature selection in sklearn

```py
# Import needed packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
```
```py
# Load the Wisconsin Breast Cancer Database
wbcd = pd.read_csv("WisconsinBreastCancerDatabase.csv")
```
```py
# Select and scale input features, create dataframe for output feature
X = wbcd[
    [
        "Radius mean",
        "Texture mean",
        "Area mean",
        "Smoothness mean",
        "Compactness mean",
        "Concavity mean",
        "Concave points mean",
        "Fractal dimension mean",
        "Symmetry mean",
    ]
]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y = wbcd[["Diagnosis"]]
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=123
)
```
```py
# Perform feature selection using the SelectKBest function
model_kbest = SelectKBest(score_func=f_classif, k=5)
X_new_kbest = model_kbest.fit_transform(X_train, np.ravel(y_train))

# Perform feature selection using the SelectPercentile function
model_percent = SelectPercentile(score_func=f_classif, percentile=30)
X_new_percent = model_percent.fit_transform(X_train, np.ravel(y_train))
```
```py
# Get features selected by each function
filter_kbest = model_kbest.get_support()
filter_percent = model_percent.get_support()

# Get input feature names
features = np.array(X_train.columns)
```
```py
# Display feature names selected by the SelectKBest function
features[filter_kbest]
```
```py
# Display feature names selected by the SelectPercent function
features[filter_percent]
```
```py
# Display the F-statistic and p-value for each feature
data = {"F-statistic": model_kbest.scores_, "p-value": model_kbest.pvalues_}
pd.DataFrame(data, index=X_train.columns)
```
```py
# Construct MLP classifier using all features and display classification accuracy
clf = MLPClassifier(random_state=1, max_iter=1000).fit(X_train, np.ravel(y_train))
clf.score(X_test, y_test)
```
```py
# Construct MLP classifier using 5 best features and display classification accuracy
clf_reduced_kbest = MLPClassifier(random_state=1, max_iter=1000).fit(
    X_train[features[filter_kbest]], np.ravel(y_train)
)
clf_reduced_kbest.score(X_test[features[filter_kbest]], y_test)
```
```py
# Construct MLP classifier using the top 30% features and display classification accuracy
clf_reduced_percent = MLPClassifier(random_state=1, max_iter=1000).fit(
    X_train[features[filter_percent]], np.ravel(y_train)
)
clf_reduced_percent.score(X_test[features[filter_percent]], y_test)
```

## 11.1.2: Wrapper methods in sklearn

```py
# Import needed packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.pipeline import Pipeline
```
```py
# Load the Wisconsin Breast Cancer Database
wbcd = pd.read_csv("WisconsinBreastCancerDatabase.csv")
```
```py
# Select and scale input features, create dataframe for output feature
X = wbcd[
    [
        "Radius mean",
        "Texture mean",
        "Area mean",
        "Smoothness mean",
        "Compactness mean",
        "Concavity mean",
        "Concave points mean",
        "Fractal dimension mean",
        "Symmetry mean",
    ]
]
y = wbcd[["Diagnosis"]]
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
```
```py
# Construct a scaler
scaler = StandardScaler()
```
```py
# Construct an estimator
estimator = SVC(kernel="linear")
```
```py
# Construct a recursive feature eliminator
rfe = RFE(estimator, n_features_to_select=5, step=1)
```
```py
# Construct a pipeline that scales the data and performs RFE
pipe_rfe = Pipeline(steps=[("scaler", scaler), ("rfe", rfe)])
```
```py
# Fit the model at the end of the pipeline using the training set
pipe_rfe.fit(X_train, np.ravel(y_train))
```
```py
# Display the selected features
X.columns[pipe_rfe["rfe"].support_]
```
```py
# Display classification score
pipe_rfe.score(X_test, y_test)
```
```py
# Construct a recursive feature eliminator with cross-validation
rfecv = RFECV(estimator, cv=4, step=1)
```
```py
# Construct a pipeline that scales the data and performs RFECV
pipe_rfecv = Pipeline(steps=[("scaler", scaler), ("rfecv", rfecv)])
```
```py
# Fit the model at the end of the pipeline using the training set
pipe_rfecv.fit(X_train, np.ravel(y_train))
```
```py
# Display the selected features
X.columns[pipe_rfecv["rfecv"].support_]
```
```py
# Display classification score
pipe_rfecv.score(X_test, y_test)
```
```py
# Construct a backward sequential feature selector
sfs = SequentialFeatureSelector(estimator, direction="backward", cv=10)
```
```py
# Construct a pipeline that scales the data and performs forward SFS
pipe_sfs = Pipeline(steps=[("scaler", scaler), ("sfs", sfs), ("model", estimator)])
```
```py
# Fit the model at the end of the pipeline using the training set
pipe_sfs.fit(X_train, np.ravel(y_train))
```
```py
# Display the selected features
X.columns[pipe_sfs["sfs"].support_]
```
```py
# Display classification score
pipe_sfs.score(X_test, y_test)
```

## 11.1 Challenge: Feature selection in Python

### Level 1

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

* Perform feature selection by initializing and fitting a `SelectKBest()` function, with `score_func=f_classif` and `k=6`.
* Get the features selected by the function.

The code provided contains all imports, loads the dataset, creates a dataframe with the standardized input features, splits the data into test and train sets, and prints the features selected by the function.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dry beans dataset
bean = pd.read_csv("Dry_Bean_Dataset.csv")

# Define input and output features
X = bean.drop(["Class"], axis=1)
y = bean[["Class"]]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

# --------------------------------------
# Initialize a SelectKBest() function with score_func=f_classif and k=6
beanKBest = SelectKBest(score_func=f_classif, k=6)

# Fit and transform the function
newX = beanKBest.fit_transform(X_train, np.ravel(y_train))

# Get the features selected by the function
featureFilter = beanKBest.get_support()
# --------------------------------------

# Get input feature names
features = np.array(X_train.columns)

# Print feature names selected by the SelectKBest() function 
print(features[featureFilter])
```

### Level 2

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Construct a function to perform recursive feature elimination with cross-validation with `cv=7` using the initialized estimator.
- Construct a pipeline that scales the data and performs `RFECV`.
- Fit the model at the end of the pipeline using the training set.

The code provided contains all imports, loads the dataset, creates a dataframe with the input features, splits the data into test and train sets, constructs a scaler, initializes an estimator function, and prints the classification score.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

# Load the dry beans dataset
bean = pd.read_csv("Dry_Bean_Dataset.csv")

# Define input and output features
X = bean.drop(["Class"], axis=1)
y = bean[["Class"]]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Construct a scaler
scalerFeat = StandardScaler()

# Construct an estimator
beanEstimator = LinearDiscriminantAnalysis()

# -------------------------------------------
# Perform recursive feature elimination with cross-validation with cv=7 using the given estimator
rfecvBean = RFECV(beanEstimator, cv=4)

# Construct a pipeline that scales the data and performs RFECV
beanRFECVPipe = Pipeline(steps=[("scaler", scalerFeat), ("rfecv", rfecvBean)]) 

# Fit the model at the end of the pipeline using the training set
beanRFECVPipe.fit(X_train, np.ravel(y_train))
# -------------------------------------------

# Print classification score
print(beanRFECVPipe.score(X_test, y_test))
```

### Level 3

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

* Construct a forward sequential feature selector with `n_features_to_select="auto"` and `tol=None` using the initialized estimator.
* Construct a pipeline that scales the data and performs forward SFS with a linear discriminant analysis model.
* Fit the model at the end of the pipeline using the training set.

The code provided contains all imports, loads the dataset, creates a dataframe with the input features, splits the data into test and train sets, constructs a scaler, initializes an estimator function, and prints the features selected by the model and the classification score.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

# Load the dry beans dataset
df = pd.read_csv("Dry_Bean_Dataset.csv")

# Define input and output features
X = df.drop(["Class"], axis=1)
y = df[["Class"]]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Construct a scaler
scalerFeat = StandardScaler()

# Initialize a linear discriminant analysis estimator
ldaEstimator = LinearDiscriminantAnalysis()

# -----------------------------------
# Construct a forward sequential feature selector using the initialized estimator
sfsBean = SequentialFeatureSelector(ldaEstimator, direction="forward", n_features_to_select="auto", tol=None)

# Construct a pipeline that scales the data and performs forward SFS and linear discriminant analysis
beanSFSPipe = Pipeline(steps=[("scaler", scalerFeat), ("sfs", sfsBean), ("model", ldaEstimator)])

# Fit the model at the end of the pipeline using the training set
beanSFSPipe.fit(X_train, np.ravel(y_train))
# -----------------------------------

# Print the selected features 
print(X.columns[beanSFSPipe.named_steps["sfs"].support_])

# Print classification score
print(beanSFSPipe.score(X_test, y_test))
```