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