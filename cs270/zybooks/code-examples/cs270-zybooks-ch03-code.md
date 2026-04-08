# CS 270 Zybooks Chapter 3 code examples

## 3.2: Classification Metrics

### Confusion mtx of Gaussian Naive Bayes model

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the hawks dataset
hawks = pd.read_csv("hawks.csv")

# Define input and output features
X = hawks[["Hallux", "Culmen"]]
y = hawks[["Species"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a Gaussian Naive Bayes model
NBModel = GaussianNB()
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
predNB = NBModel.predict(X)

# Calculate the confusion matrix 
confMatrix = metrics.confusion_matrix(np.ravel(y), predNB)

# Print the confusion matrix of the Gaussian Naive Bayes model
print("GaussianNB model\n", confMatrix)
```

### Accuracy & Kappa of Gaussian Naive Bayes model

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the hawks dataset
hawks = pd.read_csv("hawks.csv")

# Define input and output features
X = hawks[["Tail", "Weight"]]
y = hawks[["Species"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a Gaussian Naive Bayes model
NBModel = GaussianNB()
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
hawkPred = NBModel.predict(X)

# Calculate the accuracy
accuracy = metrics.accuracy_score(np.ravel(y), hawkPred)

# Calculate kappa
kappa = metrics.cohen_kappa_score(np.ravel(y), hawkPred)

# Print accuracy and kappa of the Gaussian Naive Bayes model
print("GaussianNB model accuracy:", round(accuracy, 3))
print("GaussianNB model kappa:", round(kappa, 3))
```

### Precision & Recall of Gaussian Naive Bayes model

```py
# Import packages and functions
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the field goal dataset
fieldGoal = pd.read_csv("fg_attempt.csv")

# Define input and output features
X = fieldGoal[["Distance", "ScoreDiffPreKick"]]
y = fieldGoal[["Outcome"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a Gaussian Naive Bayes model
NBModel = GaussianNB()
NBModel.fit(X, np.ravel(y))

# Calculate the predictions for each instance in X
predNB = NBModel.predict(X)

# Calculate the precision
precision = metrics.precision_score(np.ravel(y), predNB)

# Calculate the recall
recall = metrics.recall_score(np.ravel(y), predNB)

# Print precision and recall of the Gaussian Naive Bayes model
print("GaussianNB model precision:", round(precision, 3))
print("GaussianNB model recall:", round(recall, 3))
```

## 3.3: Plotting


### Classification models: Confusion mtx, Decision boundary, & ROC curve

```py
# imports (I just copied & pasted this from the file, idk how many of these are used in the excerpt I have here)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from mlxtend.plotting import plot_decision_regions

import warnings
warnings.simplefilter("ignore")

# Load the diabetes dataset
diabetes = pd.read_csv("diabetesNHANES.csv")

# Take a random sample of 100 rows
df = diabetes.sample(n=100, random_state=1234)
df.head()

# Define input features and output features
X = df[["glucose", "cholesterol"]]
y = df[["outcome"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a LDA model
ldaModel = LinearDiscriminantAnalysis()
ldaModel.fit(X, np.ravel(y))

# Confusion matrix plot for ldaModel
ConfusionMatrixDisplay.from_estimator(ldaModel, X, y)
plt.show()

# Decision boundary plot for ldaModel
plot_decision_regions(X, np.ravel(y), clf=ldaModel)
plt.show()


# Initialize and fit a Gaussian naive Bayes model
gnbModel = GaussianNB()
gnbModel.fit(X, np.ravel(y))

# ROC curves for both ldaModel and gnbModel:

# ROC curve for ldaModel
lda_plot = RocCurveDisplay.from_estimator(ldaModel, X, y, linewidth=3)
# ROC curve for gnbModel, added to lda_plot by adding ax=lda_plot.ax_
gnb_plot = RocCurveDisplay.from_estimator(
    gnbModel, X, y, ax=lda_plot.ax_, linewidth=1.5
)
plt.show()
```

### Regression models: Prediction error plot & Partial dependence plot

```py
# imports (idk if they're all used in the excerpt I've included)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# Load the diabetes dataset
diabetes = pd.read_csv("diabetesNHANES.csv")

# take a random sample of 100 rows
df = diabetes.sample(n=100, random_state=4321)

# Define input features and output features
X = df[["glucose", "cholesterol", "systolic"]]
y = df[["insulin"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and fit a k-nearest neighbors model with k=3
KNModel = KNeighborsRegressor(n_neighbors=3)
KNModel.fit(X, y)

# Generate a prediction error plot:

# - Get predictions
KNPred = KNModel.predict(X)

# - Compute prediction errors
KNPredError = y - KNPred

# - Plot prediction errors vs predicted values
fig = plt.figure()
plt.scatter(KNPred, KNPredError)
plt.xlabel("Predicted")
plt.ylabel("Prediction error")

# - Add dashed line at y=0
plt.plot([min(KNPred) - 2, max(KNPred) + 2], [0, 0], linestyle="dashed", color="black")
plt.show()

# Generate a parial dependence display for all three input features
# Add feature_names parameter to specify the plot labels
PartialDependenceDisplay.from_estimator(
    KNModel, X, features=[0, 1, 2], feature_names=["Glucose", "Cholesterol", "Systolic"]
)
plt.show()
```

## 3.6 LAB


TODO: copy lab instructions here.

```py
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Silence warning from sklearn
import warnings
warnings.simplefilter("ignore")

# Input the random state
rand = int(input())

# Load sample set by a user-defined random state into a dataframe. 
NBA = pd.read_csv("nbaallelo_log.csv").sample(n=500, random_state=rand)

# Create binary feature for game_result with 0 for L and 1 for W
NBA["win"] = NBA["game_result"].replace(to_replace = ["L","W"], value = [int(0), int(1)])

# Store relevant columns as variables
X = NBA[["elo_i"]]
y = NBA[["win"]]

# Build logistic model with default parameters, fit to X and y
model = LogisticRegression()
y_true = np.ravel(y) # IMPORTANT: the input to all the metric functions MUST be arrays
model.fit(X, y_true)

# Use the model to predict the classification of instances in X
logPredY = model.predict(X)

# Calculate the confusion matrix for the model
confMatrix = metrics.confusion_matrix(y_true, logPredY)
print("Confusion matrix:\n", confMatrix)

# Calculate the accuracy for the model
accuracy = metrics.accuracy_score(y_true, logPredY)
print("Accuracy:", round(accuracy,3))

# Calculate the precision for the model
precision = metrics.precision_score(y_true, logPredY)
print("Precision:", round(precision,3))

# Calculate the recall for the model
recall = metrics.recall_score(y_true, logPredY)
print("Recall:", round(recall, 3))

# Calculate kappa for the model
kappa = metrics.cohen_kappa_score(y_true, logPredY)
print("Kappa:", round(kappa, 3))
```

## 3.7 LAB

TODO: copy the instructions here.

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics

# Load diamonds sample into dataframe
diamonds = pd.read_csv("diamonds.csv").sample(n=50, random_state=42)

# Get user-input features
feature1 = input()
feature2 = input()

# Define input and output features
X = diamonds[[feature1, feature2]]
y = diamonds["price"]

y_true = np.ravel(y)
# Initialize and fit a multiple linear regression model
model = LinearRegression()
model.fit(X, y_true)

# Use the model to predict the classification of instances in X
mlrPredY = model.predict(X)

# Compute prediction errors
mlrPredError = y_true - mlrPredY

# Plot prediction errors vs predicted values. Label the x-axis as 'Predicted' and the y-axis as 'Prediction error'
scatterfig = plt.figure()
plt.scatter(mlrPredY, mlrPredError)
plt.xlabel('Predicted')
plt.ylabel('Prediction error')

# Add dashed line at y=0
plt.plot([min(mlrPredY)-2, max(mlrPredY)+2], [0,0], linestyle='dashed', color='black')
scatterfig.savefig("predictionError.png")
# plt.show()

# Generate a partial dependence display for both input features
display = PartialDependenceDisplay.from_estimator(
    model, X, features=[0,1], feature_names=[feature1, feature2]
)
display.figure_.savefig("partialDependence.png")
# plt.show()



# Calculate mean absolute error for the model
mae = metrics.mean_absolute_error(y, mlrPredY)
print("MAE:", round(mae, 3))
```
