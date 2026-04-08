# CS 270 Ch. 6 Code Examples

## 6.1: Decision Tree Classifiers

### 6.1.1: DTCs in sklearn

```py
import warnings
warnings.simplefilter("ignore")
```
```py
# Import needed packages and functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```
```py
# Import the data
Coffee = pd.read_csv("coffee.csv")

# Filter out the coffee that is missing scores
Coffee = Coffee[Coffee["total_cup_points"] != 0]
Coffee.shape
```
```py
# Summarize the numerical features
Coffee.describe()
```
```py
# Summarize the categorical features
Coffee.describe(include=["O"])
```
```py
# Select features
X = Coffee[["uniformity", "sweetness"]]
y = pd.get_dummies(Coffee["species"], drop_first=True)
# Dummy variables are not required for DecisionTreeClassifier but are required for
# plot_decision_boundary

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)
```
```py
# Scatter plot of the input features colored by species of coffee
p1 = sns.scatterplot(
    data=X_train, x="uniformity", y="sweetness", hue=y_train["Robusta"], alpha=0.5
)
p1.set(xlabel="Uniformity", ylabel="Sweetness")
p1.get_legend().set_title("Species")
```
```py
# Initialize the tree and fit the tree
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
```
```py
# Plot the fitted tree
plot_tree(DTC, feature_names=X_train.columns)
```
```py
# Plot the decision boundary

ax = plot_decision_regions(X_train.to_numpy(), np.ravel(y_train).astype(int), clf=DTC, legend=0)
plt.xlabel("Uniformity")
plt.ylabel("Sweetness")


# Change the labels to species names
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, ["Arabica", "Robusta"], framealpha=0.3, scatterpoints=1)
```
```py
# Predict for the test set and plot the confusion matrix
y_pred = DTC.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
```
```py
# Predict the probabilities
# (since the tree's leaves are all pure, the values are all 0 and 1)
DTC.predict_proba(X_train)
```

### 6.1.2: `DecisionTreeClassifier()` parameters

```py
# Import needed packages and functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```
```py
# Import the data
Coffee = pd.read_csv("coffee.csv")

# Filter out the coffee that is missing scores
Coffee = Coffee[Coffee["total_cup_points"] != 0]
Coffee.shape
```
```py
# Summarize the numerical values
Coffee.describe()
```
```py
# Summarize the categorical features
Coffee.describe(include=["O"])
```
```py
# Select features
X = Coffee[
    [
        "aroma",
        "flavor",
        "aftertaste",
        "acidity",
        "body",
        "balance",
        "uniformity",
        "clean_cup",
        "cupper_points",
        "sweetness",
    ]
]
y = Coffee["species"]

# Create training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)
```
```py
# Initialize the tree and fit the tree
DTC = DecisionTreeClassifier(
    max_depth=5, 
    min_samples_split=4, 
    min_samples_leaf=3, 
    max_leaf_nodes=12
    )
DTC.fit(X_train, y_train)
```
```py
DTC.get_depth()
```
```py
# Plot the fitted tree
plot_tree(DTC, feature_names=X_train.columns)
```
```py
# Predict for the test set and plot the confusion matrix
y_pred = DTC.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
```
```py
# Predict the probabilities
DTC.predict_proba(X_train)
```

### 6.1.2 (Challenge): 1

Heart disease is one of the leading causes of death in the United States. Data from patients' medical records can be used to develop early diagnostic tools for heart disease that may save thousands of lives every year.

The heart dataset contains data from a sample of medical records in the United States, Hungary, and Switzerland. For each patient, `target=0` if the patient does not have heart disease and `target=1` if the patient does have heart disease.

- Initialize a decision tree classifier model `heartDTC` using entropy as the impurity measure.

The code provided imports the dataset and packages, creates input and output feature sets, and prints the parameters of `heartDTC`.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the heart dataset
heart = pd.read_csv("heart.csv")

# Create a dataframe X containing input features maximum heart rate and cholesterol
X = heart[["thalach", "chol"]]

# Create a dataframe y containing output feature target
y = heart[["target"]]

# -----------------------------------------
# Initialize the model
heartDTC = DecisionTreeClassifier(criterion='entropy')
# -----------------------------------------

# Print model parameters
print(heartDTC.get_params())
```

### 6.1.2 (Challenge): 2

Heart disease is one of the leading causes of death in the United States. Developing early diagnostic tools for heart disease may save thousands of lives every year.

The heart dataset contains data from a sample of medical records in the United States, Hungary, and Switzerland. For each patient, `target=0` if the patient does not have heart disease and `target=1` if the patient does have heart disease.

- Initialize a decision tree classifier model `DTCHeart` using Gini impurity as the impurity measure.
- Fit `DTCHeart` to input features X and output feature y.

The code provided imports the dataset and packages, scales the input features, and prints the predictions and proportion of correctly predicted instances from the fitted model.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the heart dataset
heart = pd.read_csv("heart.csv")

# Create a dataframe X containing input features age in years and cholesterol
X = heart[["age", "chol"]]

# Create a dataframe y containing output feature target
y = heart[["target"]]

# ---------------------------------------
# Initialize the model
DTCHeart = DecisionTreeClassifier(criterion='gini')

# Fit the model to X and y
DTCHeart.fit(X, y)
# ---------------------------------------


# Print model predictions
print(DTCHeart.predict(X))

# Print proportion of instances classified correctly
print(DTCHeart.score(X,y))
```

### 6.1.2 (Challenge): 3

Heart disease is one of the leading causes of death in the United States. Developing early diagnostic tools for heart disease may save thousands of lives every year.

The heart dataset contains data from a sample of medical records in the United States, Hungary, and Switzerland. For each patient, `target=0` if the patient does not have heart disease and `target=1` if the patient does have heart disease.

- Initialize a decision tree classifier model `heartDTCEntropy` with `criterion="entropy"`, `max_depth=3`, and `random_state=123`.
- Fit `heartDTCGini` and `heartDTCEntropy` to input features X and output feature y.
- Print a text version of the tree from `heartDTCEntropy`.

The code provided imports the dataset and packages, initializes `heartDTCGini`, and prints the tree from `heartDTCGini`.

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# Load the heart dataset
heart = pd.read_csv("heart.csv")

# Create a dataframe X containing age in years and resting blood pressure
X = heart[["age", "trestbps"]]

# Create a dataframe y containing output feature target
y = heart[["target"]]

# Initialize the models
heartDTCGini = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=123)

# ---------------------------------------------
heartDTCEntropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=123)

# Fit both models to X and y
heartDTCGini.fit(X, y)
heartDTCEntropy.fit(X, y)
# ---------------------------------------------

# Print the tree for the Gini impurity model
print(export_text(heartDTCGini, feature_names=X.columns.to_list()))

# Print the tree for the entropy model
# ---------------------------------------------
print(export_text(heartDTCEntropy, feature_names=X.columns.to_list()))
# ---------------------------------------------
```

## 6.3: Decision Tree Regressors

```py
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
```
```py
Coffee = pd.read_csv("coffee.csv")
Coffee.columns
```
```py
X = Coffee[
    ["aroma", "body", "species"]
]  #'aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance', 'uniformity', 'clean_cup', 'sweetness', 'species']]
X_dummies = pd.get_dummies(X, drop_first=True)
y = Coffee["cupper_points"]

X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, random_state=543)
```
```py
DTR = DecisionTreeRegressor(max_depth=2)

DTR.fit(X_train, y_train)
```
```py
plot_tree(DTR, feature_names=X_train.columns)
```
```py
# Prints the r-squared value for the training set.
DTR.score(X_train, y_train)
```
```py
DTR = DecisionTreeRegressor(max_leaf_nodes=10, min_samples_leaf=20)
DTR.fit(X_train, y_train)
# Adjust plot size to make the tree more readable
plt.figure(figsize=(10, 8))
plot_tree(DTR, feature_names=X_train.columns)
plt.show()  # Suppresses the text output
```
```py
# Adjust ccp_alpha to improve how well the tree generalizes to the test set.
DTR = DecisionTreeRegressor(ccp_alpha=0.000)
DTR.fit(X_train, y_train)
DTR.score(X_test, y_test)
```
```py
# Adjust plot size as needed
plt.figure(figsize=(12,10))

plot_tree(DTR, feature_names=X_train.columns)
plt.show()
```