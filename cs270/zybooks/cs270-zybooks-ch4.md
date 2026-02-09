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

See examples [here](./code-examples/cs270-zybooks-ch4-code.md)

# 4.2: Cross-Validation for model selection

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

See code examples [here](./code-examples/cs270-zybooks-ch4-code.md).

### Final model selection

Model selection identifies the best model type&mdash;but not necessarily the best params of the best model. (e.g. KNN does not have params that must be trained, but log reg does.) 

- Models WITHOUT trained params can be EVALUATED DIRECTLY on the testing set. 
- Models WITH trained params MUST BE FITTED to the entire training set BEFORE EVALUATING on testing.
  - Note that while you trained on only part of the training set during CV model selection, when you select your model you fit it to the ENTIRE TRAINING SET before testing.
    - (This avoids selection bias.)
  - Do not select the best params from a single CV fold. That introduces (selection?) bias.

### Examples

See code examples [here](./code-examples/cs270-zybooks-ch4-code.md).

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

See examples for `GridSearchCV()` [here](./code-examples/cs270-zybooks-ch4-code.md)

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

See examples for `validation_curve()` [here](./code-examples/cs270-zybooks-ch4-code.md).

## Examples

See more examples [here](./code-examples/cs270-zybooks-ch4-code.md)

# 4.4 & 4.5 LABS

See labs [here](./code-examples/cs270-zybooks-ch4-code.md)
