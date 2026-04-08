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
