# CS 270 Ch. 11 Code Examples

## 11.1

### 11.1.1: Filter-based feature selection in sklearn

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

### 11.1.2: Wrapper methods in sklearn

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

### 11.1 Challenge: Feature selection in Python

#### Level 1

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

#### Level 2

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

#### Level 3

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

## 11.2

### 11.2.1: PCA in sklearn

```py
# Import required libraries
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```
```py
# Load a subset with 1000 instances of the MNIST digits dataset
digits = pd.read_csv("https://static-resources.zybooks.com/MachineLearning/digits.csv")
digits_sample = digits.sample(7000, random_state=246)
```
```py
# Subset input and output features
X = digits_sample.iloc[:, :-1]
y = digits_sample[["class"]]
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234
)
```
```py
# Create a pipeline that fits an SVC model to training data
clf = SVC(class_weight="balanced")
clf.fit(X_train, np.ravel(y_train))
```
```py
# Display accuracy of the classifier using the test set
clf.score(X_test, y_test)
```
```py
# Create a principal component analysis model with 64 components
pca_64 = PCA(n_components=64)
scaler = StandardScaler()
pipeline_pca_64 = Pipeline(steps=[("scaler", scaler), ("pca_64", pca_64), ("clf", clf)])
pipeline_pca_64.fit(X_train, np.ravel(y_train))
```
```py
# Scree plot
plt.plot(pca_64.explained_variance_)
plt.xlabel("Components", size=14)
plt.ylabel("Explained variance", size=14)
```
```py
# Create a principal component analysis model with 20 components
pca_20 = PCA(n_components=20)
pipeline_pca_20 = Pipeline(steps=[("scaler", scaler), ("pca_20", pca_20), ("clf", clf)])
pipeline_pca_20.fit(X_train, np.ravel(y_train))
```
```py
# Display accuracy of the classifier using the 20 principal components
pipeline_pca_20.score(X_test, y_test)
```
```py
# Visualize the covariance matrix using a heatmap
fig, ax = plt.subplots(1)
p = plt.imshow(pca_20.get_covariance(), cmap="viridis")
fig.colorbar(p, ax=ax)
plt.show()
```
```py
# Display covariance matrix
np.round(pd.DataFrame(pca_20.get_covariance()), 4)
```
```py
# Display the original image for an instance
image_original = X_train.iloc[4, :].to_numpy()
image_original = image_original.reshape([28, 28])
plt.imshow(image_original, cmap=plt.cm.gray_r)
```
```py
# Display the transformed image of the same instance
digits_pca_reduced = pca_20.fit_transform(X_train)
digits_pca_recovered = pca_20.inverse_transform(digits_pca_reduced)
image_pca = digits_pca_recovered[4, :].reshape([28, 28])
plt.imshow(image_pca, cmap=plt.cm.gray_r)
```
```py
# Display the principal components
pd.DataFrame(np.round(pca_20.components_, 4))
```
```py
# Display the amount of variance explained by the principal components
np.round(pca_20.explained_variance_, 4)
```
```py
# Display the percentage variance explained by the principal components
np.round(pca_20.explained_variance_ratio_, 4)
```

### 11.2.2: ICA in sklearn

```py
# Import required libraries
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```
```py
# Randomly generate original sources
np.random.seed(123)

samples = 2000
time = np.linspace(0, 10, samples)

signal_1 = np.cos(2 * time)
signal_2 = signal.sawtooth(3 * np.pi * time)

S = np.c_[signal_1, signal_2]
```
```py
# Specify mixing matrix and compute mixed signals
A = np.array([[0.5, 3], [0.25, 2]])
X = np.dot(S, A.T)
```
```py
# Create a FastICA model
ica = FastICA(n_components=2)
```
```py
# Fit and transform the mixed signals using the FastICA model
S_ = ica.fit_transform(X)
```
```py
# Graph the original sources, mixed, signals, estimated sources
graphs = [S, X, S_]
titles = [
    "Original sources",
    "Mixed signals",
    "Estimated sources",
]
colors = ["lightskyblue", "orange"]

for ii, (graph, title) in enumerate(zip(graphs, titles), 1):
    plt.subplot(4, 1, ii)
    plt.title(title)
    for sig, color in zip(graph.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
```
```py
# Display the estimated mixing matrix
ica.mixing_
```
```py
# Display the unmixing matrix
ica.components_
```
```py
# Display the whitening matrix
ica.whitening_
```

### 11.2.3: FA in sklearn

```py
# Import required libraries
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import seaborn as sns
```
```py
# Load the airline passenger satisfcation dataset
passengers = pd.read_csv("passengers.csv")
```
```py
# Subset input and output features
X = passengers.iloc[:, :-1]
y = passengers[["Satisfied"]]
```
```py
# Display columns
X.columns
```
```py
# Display first five instances
X.head()
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
```
```py
# Create data pipeline with the scaler, factor analysis model, and support vector classifier
scaler = StandardScaler()
factor_analysis = FactorAnalysis(rotation="varimax", n_components=6, random_state=123)
clf = SVC(random_state=123)
pipeline_factor_analysis = Pipeline(
    steps=[("scaler", scaler), ("factor_analysis", factor_analysis), ("clf", clf)]
)
```
```py
# Fit the data pipeline to the training set
pipeline_factor_analysis.fit(X_train, np.ravel(y_train))
```
```py
# Display accuracy of the classifier using the test set
pipeline_factor_analysis.score(X_test, y_test)
```
```py
# Create another data pipeline using the FactorAnalyzer() function
factor_analyzer = FactorAnalyzer(rotation="varimax", n_factors=6)
pipeline_factor_analyzer = Pipeline(
    steps=[("scaler", scaler), ("factor_analyzer", factor_analyzer), ("clf", clf)]
)
```
```py
# Fit the data pipeline that uses FactorAnalyzer to the training set
pipeline_factor_analyzer.fit(X_train, np.ravel(y_train))
```
```py
# Display accuracy of the classifier using the test set
pipeline_factor_analyzer.score(X_test, y_test)
```
```py
# Create a matrix of factor loadings
loadings = pd.DataFrame(
    factor_analyzer.loadings_,
    columns=["FA1", "FA2", "FA3", "FA4", "FA5", "FA6"],
    index=X.columns,
)
```
```py
# Display the factor loadings
np.round(loadings, 4)
```

### 11.2 Challenge activity: Feature extraction using linear techniques in Python

#### Level 1 (PCA)

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Create a principal component analysis model using the `PCA()` function with 6 components and `random_state=39`.
- Create a data pipeline with a scaler, pca model, and a support vector classifier.
- Fit the model at the end of the pipeline using the training set.
- Get the principal components.

The code provided contains all imports, loads the dataset, creates dataframes with the input and output features, splits the data into test and train sets, initializes a support vector classifier, initializes a scaler, and prints the accuracy of the classifier using the 6 principal components and the principal components.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
from sklearn.svm import SVC
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Load the dry beans dataset
bean = pd.read_csv("Dry_Bean_Dataset.csv")

# Define input and output features
X = bean.drop(["Class"], axis=1)
y = bean[["Class"]]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create a support vector classifier
clf = SVC(gamma="scale", class_weight="balanced", C=50, random_state=123)

# Create a scaler to standardize input features
scaler = StandardScaler()

# -------------------------------
# Create a principal component analysis model using the PCA() function with 6 
# components and random_state=39
beanPCA = PCA(n_components=6, random_state=39)

# Create a data pipeline with a scaler, pca model, and a support vector classifier
pipelineBean = Pipeline(steps=[("scaler", scaler), ("pca", beanPCA), ("clf", clf)])

# Fit the model at the end of the pipeline using the training set
pipelineBean.fit(X_train, y_train)

# Get the principal components
beanComponents = beanPCA.components_
# -------------------------------

# Print the principal components and accuracy
print(beanComponents)
print("Accuracy is:", pipelineBean.score(X_test, y_test))
```

#### Level 2 (ICA)

- Create an ICA model using the `FastICA()` function with 3 components, `algorithm="deflation"`, and `whiten="unit-variance"`.
- Fit and transform the mixed signals using the FastICA model.
- Get the estimated mixing matrix.

The code provided contains all imports, generates the original sources, creates a mixing matrix, computes the mixed signals, and prints the first five elements of the estimated source signal and the estimated mixing matrix.

```py
# Import packages and functions
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Randomly generate original sources
np.random.seed(19)

samples = 2000
time = np.linspace(0, 10, samples)

signal_1 = np.cos(time)
signal_2 = signal.square(3*np.pi*time) 
signal_3 = signal.sawtooth(2*np.pi*time) 

S = np.c_[signal_1, signal_2, signal_3]

# Specify mixing matrix and compute mixed signals
A = np.array([[2, 0.4, 1], [3, 0.1, 0.5], [0.4, 2, 3]])
X = np.dot(S, A.T)

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# Create a FastICA model with three components, algorithm="deflation", and whiten="unit-variance"
independentCA = FastICA(n_components=3, algorithm="deflation", whiten="unit-variance")

# Fit and transform the mixed signals using the FastICA model
estimatedS = independentCA.fit_transform(X)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Print the first five elements of the estimated source signal
print(estimatedS[0:5])

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# Get the estimated mixing matrix
mixingEstimate = independentCA.mixing_
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Print estimated mixing matrix
print(mixingEstimate)
```

#### Level 3 (FA)

Two researchers collected high-resolution images of 13,611 grains of seven varieties of dry beans. Using a computer vision process, 16 features of each bean were extracted.

- Create a factor analysis model using the `FactorAnalysis()` function with `rotation="varimax"`, 4 components, and `random_state=66`.
- Create a data pipeline with a scaler, factor analysis model, and a support vector classifier.
- Fit the model at the end of the pipeline using the training set.
- Get the noise variance for each feature.

The code provided contains all imports, loads the dataset, creates dataframes with the input and output features, splits the data into test and train sets, initializes a support vector classifier, initializes a scaler, and prints the accuracy of the classifier and the noise variance for each feature.

```py
# Import packages and functions
import warnings
warnings.simplefilter("ignore")
from sklearn.svm import SVC
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Load the dry beans dataset
bean = pd.read_csv("Dry_Bean_Dataset.csv")

# Define input and output features
X = bean.drop(["Class"], axis=1)
y = bean[["Class"]]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create a support vector classifier
clf = SVC(gamma="scale", class_weight="balanced", C=50, random_state=123)

# Create a scaler to standardize input features
scaler = StandardScaler()

# ------------------------------------
# Create a factor analysis model using the FactorAnalysis() function with 4 
# components, rotation="varimax" and random_state=66
beanFA = FactorAnalysis(rotation="varimax", n_components=4, random_state=66)

# Create a data pipeline with a scaler, factor analysis model, and a support vector classifier
faPipeline = Pipeline(steps=[("scaler", scaler), ("factor_analyzer", beanFA), ("clf", clf)])

# Fit the model at the end of the pipeline using the training set
faPipeline.fit(X_train, y_train)

# Get the noise variance for each feature
varianceNoise = beanFA.noise_variance_
# ------------------------------------

# Print the noise variance for each feature and accuracy
print(varianceNoise[:100])
print("Accuracy is:", faPipeline.score(X_test, y_test))
```

## 11.3

### 11.3.1: MDS in sklearn

```py
# Load the necessary libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
```
```py
# Load a subset with 1500 instances of the MNIST digits dataset
digits = pd.read_csv("https://static-resources.zybooks.com/MachineLearning/digits.csv")
digits_sample = digits.sample(1500, random_state=1234)
```
```py
# Subset input and output features
X = digits_sample.iloc[:, :-1]
y = digits_sample[["class"]]
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
```
```py
# Build a pipeline that fits an PCA model to the scaled training data
scaler = MinMaxScaler()
pca = PCA(n_components=2, random_state=123)
pipeline_pca = Pipeline(steps=[("scaler", scaler), ("pca", pca)])
X_transformed = pipeline_pca.fit_transform(X_train, np.ravel(y_train))
X_transformed
```
```py
# Plot the PCA mapping
fig, plot = plt.subplots()
scatter = plot.scatter(
    x=X_transformed[:, 0], y=X_transformed[:, 1], c=np.ravel(y_train), cmap="viridis"
)
plt.legend(
    bbox_to_anchor=(1, 1),
    handles=scatter.legend_elements()[0],
    labels=scatter.legend_elements()[1],
)
```
```py
# Build a pipeline that fits an MDS model to the scaled training data
mds = MDS(n_components=2, random_state=1234)
pipeline_mds = Pipeline(steps=[("scaler", scaler), ("mds", mds)])
X_transform_mds = pipeline_mds.fit_transform(X_train)
```
```py
# Plot the MDS mapping
fig, plot = plt.subplots()
scatter = plot.scatter(
    X_transform_mds[:, 0], X_transform_mds[:, 1], c=np.ravel(y_train), cmap="viridis"
)
plt.legend(
    bbox_to_anchor=(1, 1),
    handles=scatter.legend_elements()[0],
    labels=scatter.legend_elements()[1],
)
```
```py
# Display the data points in lower-dimensional space
pd.DataFrame(mds.embedding_)
```
```py
# Display the dissimilarity matrix in the higher-dimensional space
pd.DataFrame(mds.dissimilarity_matrix_)
```
```py
# Display the raw stress
print(mds.stress_)
```
```py
# Display the number of iterations that give the best stress
mds.n_iter_
```

### 11.3.2: Isomap in sklearn

```py
# Load the necessary libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler
```
```py
# Load a subset with 1500 instances of the MNIST digits dataset
digits = pd.read_csv("https://static-resources.zybooks.com/MachineLearning/digits.csv")
digits_sample = digits.sample(1500, random_state=1234)
```
```py
# Subset input and output features
X = digits_sample.iloc[:, :-1]
y = digits_sample[["class"]]
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
```
```py
# Build a pipeline that fits an PCA model to the scaled (normalized) training data
scaler = MinMaxScaler()
pca = PCA(n_components=2, random_state=123)
pipeline_pca = Pipeline(steps=[("scaler", scaler), ("pca", pca)])
X_transformed = pipeline_pca.fit_transform(X_train, np.ravel(y_train))
```
```py
# Plot the PCA mapping
fig, plot = plt.subplots()
scatter = plot.scatter(
    X_transformed[:, 0], X_transformed[:, 1], c=np.ravel(y_train), cmap="viridis"
)
plt.legend(
    bbox_to_anchor=(1, 1),
    handles=scatter.legend_elements()[0],
    labels=scatter.legend_elements()[1],
)
```
```py
# Build a pipeline that fits an Isomap model to the scaled (normalized) training data
isomap = Isomap(n_components=2, n_neighbors=25)
pipeline_isomap = Pipeline(steps=[("scaler", scaler), ("isomap", isomap)])
X_transform_isomap = pipeline_isomap.fit_transform(X_train)
```
```py
# Plot the Isomap mapping
fig, plot = plt.subplots()
scatter = plot.scatter(
    X_transform_isomap[:, 0],
    X_transform_isomap[:, 1],
    c=np.ravel(y_train),
    cmap="viridis",
)
plt.legend(
    bbox_to_anchor=(1, 1),
    handles=scatter.legend_elements()[0],
    labels=scatter.legend_elements()[1],
)
```
```py
# Display the data points in lower-dimensional space
pd.DataFrame(isomap.embedding_)
```
```py
# Display the KernelPCA object used in the mapping
isomap.kernel_pca_
```
```py
# Display the geodesic distances in higher-dimensional space
isomap.dist_matrix_
```
```py
# Display the nearest neighbors object used in the mapping
isomap.nbrs_
```

### 11.3.3: t-SNE in sklearn

```py
# Load the necessary libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
```
```py
# Load a subset with 1500 instances of the MNIST digits dataset
digits = pd.read_csv("https://static-resources.zybooks.com/MachineLearning/digits.csv")
digits_sample = digits.sample(1500, random_state=1234)
```
```py
# Subset input and output features
X = digits_sample.iloc[:, :-1]
y = digits_sample[["class"]]
```
```py
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
```
```py
# Build a pipeline that fits an PCA model to the scaled (normalized) training data
scaler = MinMaxScaler()
pca = PCA(n_components=2, random_state=123)
pipeline_pca = Pipeline(steps=[("scaler", scaler), ("pca", pca)])
X_transformed = pipeline_pca.fit_transform(X_train, np.ravel(y_train))
```
```py
# Plot the PCA mapping
fig, plot = plt.subplots()
scatter = plot.scatter(
    X_transformed[:, 0], X_transformed[:, 1], c=np.ravel(y_train), cmap="viridis"
)
plt.legend(
    bbox_to_anchor=(1, 1),
    handles=scatter.legend_elements()[0],
    labels=scatter.legend_elements()[1],
)
```
```py
# Build a pipeline that fits a t-SNE model to the scaled (normalized) training data
tsne = TSNE(n_components=2, perplexity=45.0, random_state=123)
pipeline_tsne = Pipeline(steps=[("scaler", scaler), ("tsne", tsne)])
X_transform_tsne = pipeline_tsne.fit_transform(X_train)
```
```py
# Plot the t-SNE model mapping
X_transform_tsne = tsne.fit_transform(X_train)
fig, plot = plt.subplots()
scatter = plot.scatter(
    X_transform_tsne[:, 0], X_transform_tsne[:, 1], c=np.ravel(y_train), cmap="viridis"
)
plt.legend(
    bbox_to_anchor=(1, 1),
    handles=scatter.legend_elements()[0],
    labels=scatter.legend_elements()[1],
)
```
```py
# Display the data points in lower-dimensional space
pd.DataFrame(tsne.embedding_)
```
```py
# Display the KL divergence after optimization
tsne.kl_divergence_
```
```py
# Display the effective learning rate
tsne.learning_rate
```