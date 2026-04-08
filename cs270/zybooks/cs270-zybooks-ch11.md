# 11.1 Feature selection

## Dimensionality Reduction

- ***Dimensionality Reduction***: Process of **reducing number of features** in a dataset while preserving as much info as possible.
  - Feature **extraction**: Combining features into new features.
  - Feature **selection**: Selecting a subset of features.
- Goal of DR is to **simplify the data** representation to make the data more **manageable & interpretable**.
  - i.e., to simplify data representation while **retaining essential characteristics**.
- ^ DR also **reduces computation cost**, which is very important for some ML algorithms.

## Feature selection

- ***Feature selection***: Process of **selecting a subset of features** from a dataset that are **most relevant** for an ML task.
- Main goals:
  - Improve **model performance**,
  - Reduce **computational complexity**.
- Three broad categories:
  - ***Filter-based*** methods: **Rank input features** based on **how much they impact output**.
  - ***Wrapper*** methods: Select features by ***iteratively* building & evaluating models** on a subset of features.
  - ***Embedded*** methods: **integrate feature selection process into training** of ML model.

### Filter-based methods

- ***Filter-based methods*** of ftr sel: **Rank** each feature's importance (i.e., **how strongly it correlates to output**) based on **statistical tests**.
  - **Required stat test depends on data types of in/out features** (see table below).

|                           | Numeric output ftr              | Categorical output ftr  |  
| ------------------------- | ------------------------------- | ----------------------- |  
| **Numeric input ftr**     | $R$ (Pearson correlation coeff) | $F$-statistic           |  
| **Categorical input ftr** | $F$-statistic.                  | $x^2$-statistic         |  

#### Filter-based methods in Python

<!-- * Select eight best performing features using $\mathcal{X}^2$-test: `SelectKBest(score_func=chi2, k=8)`
* 40th percentile of best-performing features to keep using $F$-test for regression: `SelectPercentile() -->

See the [11.1.1 code example](./code-examples/cs270-zybooks-ch11-code.md#1111-filter-based-feature-selection-via-sklearn) for filter-based feature selection in sklearn.

### Wrapper-based methods

- ***Recursive feature elimination (RFE)***: **Iteratively removes the least important feature(s)** from the dataset, until a desired num. of features is reached.
  - Not always practical: **A dataset with $p$ features contains $2^p$ subsets.** Exhaustsively searching every subset is slow.
- ***Sequential feature selection (SFS)***: Selects or removes ftrs in a **step-by-step manner** by using **CV techniques** to evaluate model performance at each iteration.
  - Idrk what the "sequential" part of SFS is...
  - Two main types: 
    - ***Sequential FORWARD selection***: Starts w/ NO features, **iteratively ADDS** (starting w/ MOST important).
      - Less computationally expensive: Bc it starts w/ an empty set (rather than the entire feature space), fewer feature subsets are considered.
    - ***Sequential BACKWARD selection***: Starts w/ ALL features, **iteratively REMOVES** features (starting w/ LEAST important).
    - ^ Similar, but **don't always result in same model**.
    - ^ Both are ***monotonic***, meaning **features cannot be removed/added once added/removed**.
    - ^ (The Zybook gives algorithms/processes for both, if you care. (Algorithms 11.1.2 & 11.1.3))


#### Wrapper-based methods in Python

- sklearn:
  - `RFE()` for RFE.
  - `RFECV()` for RFE w/ CV.
  - `SequentialFeatureSelector()` for SFS (forward & backward).
- ^ `support_` attr & and `get_support()` method display the indeces of selected features.

There's a table containing the params for those, if you care. (Tables 11.1.3 & 11.1.4)

<!-- ##### `RFE()`/`RFECV()` params

| Param | Default | Description |  
| ----- | ------- | ----------- |  
| `estimator` | (None) | Model that has been fit. |  
| `step` | `1` | The number features to remove in each iteration. |  
| `importance_getter` | `"auto"` |  -->

You can also find an example of wrapper methods in use in the [code examples]() for this chapter.

### Embedded methods

- ***Embedded methods***: Selects most relevant features **when training the ML model.** They differ from ftr sel & wrapper-based methods b/c **feature selection is an integral part of the model building process**.
  - Typically a step w/in the algorithm itself, or achieve via model-specific techniques.
- Common e.g.s: tree-based methods (e.g., random forest) and regularization methods (e.g., LASSO regression).
- ADVANTAGES:
  - **Efficiency**: Feature selection is integrated into the training process. Thus it is automated and does not involve a separate step after model construction.
  - **Improved interpretability of models** (sometimes): Sometimes, embedded methods highlight which features are the most influential.
  - **Flexibility in model selection**: Along w/ selecting best features, embedded methods choose an appropriate ML algorithm. This is advantageous when exploring various modeling approaches.
- DISADVANTAGES:
  - **NOT model agnostic**. 
  - Doesn't require separate step, but cannot be separated as its own step, either.
    - Hence, it can more **computationally intensive**, b/c you have to re-train your model to get different feature selections.