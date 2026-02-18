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

- Initialize a decision tree classifier model `DTCHeart` using entropy as the impurity measure.

The code provided imports the dataset and packages, creates input and output feature sets, and prints the parameters of `DTCHeart`.