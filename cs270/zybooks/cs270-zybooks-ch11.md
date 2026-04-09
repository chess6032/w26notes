# 11.1: Feature selection

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

# 11.2: Feature extraction using linear techniques

## Feature extraction

- ***Feature extraction*** is the process of **creating new features** from raw data to **reduce dimensionality** and **identify important patterns**. 
  - (Often, raw data cannot be readily used for an ML algorithm, so data should be transformed into a format from which ftrs can be extracted.)
- Feature extraction can:
  - **improve model performance**,
  - **reduce training times**,
  - yield **btter generalization** to new data.

## Linear feature extraction

- ***Linear feature extraction*** creates new features by taking **linear combinations of existing features** instead of using all original features. 
- We'll cover a few techniques of linear feature extraction:
  - Principal component analysis (PCA)
  - Independent componenent analysis (good for "blind source separation")
  - Factor analysis

### PCA (Principle component analysis)

- ***PCA*** transforms data into a **new coordinate system** into which the **variance of the data along each axis is maximized**.
  - The axes used in PCA are called ***principal components***.
- PCA is most commonly used for:
  - Image compression
  - Noise reduction
  - Anomaly detection
- ADVANTAGES:
  - Dimensionality reduction
  - Orthgonality of principal components
  - Reduced noise
- DISADVANTAGES:
  - Loss of interpretability
  - Linearity assumption
  - Sensitivity to outliers

#### PCA process (covariance mtx!)

- ***Covariance matrix***: Contains the direction & relationships btwn features in the dataset.
  - For a dataset w/ $p$ features, its cov mtx $C$ is a $p \times p$ mtx where:
    - Off-diagonal elements contain covariances btwn two features.
    - Diagonal elements contain variances of each feature.
- ^ The cov mtx's **eigenvecs/vals** are important. In the context of a cov mtx:
  - **eigvals** indicate **amount of variance** explained by corresponding principal component.
    - **Largest eigvals capture most variance**.
  - **eigvec** describe **direction** of corresponding principle component.
- ^ In PCA, we find the cov mtx's ($C$) eigendecomposition: $C = P \Lambda P^T$
  - $P$ is a mtx whose cols are ($C$'s?) eigvecs
  - $\Lambda$ is a mtx whose diagonals are ($C$'s?) eigvals.
  - (^ WAIT I LOWK REMEMBER DOING SMTH LIKE TS FROM MATH 213 (LIN ALG).)

Here's the PCA process:

1. **Standardize the data**. (PCA is sensitive to scale. "Standardizing" data means to transform it such that it has a mean of 0 and a $\sigma$ of 1.)
2. **Compute cov mtx**. 
    - The covariance btwn ftrs $X$ and $Y$ is given in the formula below, where&mdash;
      - $x_i$ & $y_i$ are the standardized values for the $i$th instances. 
      - $\bar x$ & $\bar y$ are the means of the standardized values.
      - $N$ is num of instances


```math
\text{cov}(X, Y) = \frac {\sum (x_i - \bar x)(y_i - \bar y)} N
```

3. **Compute eigenvals/vecs** by solving $C \mathbf{v} = \lambda \mathbf{v}$ (where $\mathbf v$ is the eigvec corresponding to eigval $\lambda$).
4. **Sort & select principal components**: Largest eigvals capture most variance in data. This is often done by examining a ***scree plot*** and choosing the components that precede the "elbow".
5. **Transform the data**. The projection mtx ($P$), which uses the eigvals w/ highest eigvals as columns, transforms the original data into a lower dimensional space. The new data is obtained like this: $\text{new data} = \text{standardized data @ proj mtx}$ (where $@$ indicates matrix/vector multiplication.)

### Independent component analysis (ICA)

- ***ICA*** is a comptutational technique used to **separate multivariate signal into additive, independent components**.
  - Used frequently in signal processing, and also used in medical imaging and financial analysis.
- In contrast to PCA, the components ICA attempts to find are not necessarily orthogonal.
- ICA **can handle inputs that are nonlinear**.
- ICA is particularly good at dealing with the ***blind source separation (BSS)*** problem, which is about **separating mixed signals into original, independent sources**.
  - BSS: Consider signals $S$ whose "mixing" can be expressed as mtx multiplication with a "mixing mtx" $A$, which yields the mixed signals $X$: $X = SA$. Then your goal is to estimate an "unmixing mtx" $W$ that can reconstruct $S$.
  - Important step in using ICA to solve BSS problems is ***data whitening***, which aims to **remove correlations in the data**.
    - In data whitening, **data is rotated & normalized** such that the features:
      - are uncorrelated,
      - contain a spherical covariance structure
      - have uniform vairances.
    - (The term is called "whitening" b/c inputs are transformed into signals that resemble white noise.)
    - (BRO WTF IS A SIGNAL THEY DON'T EVEN SAY.)
- ADVANTAGES:
  - Source separation
  - Unbiased components
- DISADVANTAGES:
  - Assumption of source independence and non-Gaussianity (whatever tf *that* means)
  - Sensitivity to model & parameters

#### ICA process

1. **Mean center the data**: Subtract the mean of each ftr from the corresponding columns of $X$ (mixed signals) to ensure that the sources have **mean of zero**:

```math
X_\text{centered} = X - \bar X
```

2. **Whiten the data**: 
   1. Find the eigval decomp of the cov mtx of the centered data: $C = \frac 1 N X_\text{centered}X^T_\text{centered}=P\Lambda P^T$
       - $P$ is the mtx w/ $C$'s eigvecs as cols
       - $\Lambda$ is a diagonal mtx of $C$'s eigvals.
   1. The pre-whitening mtx is $V = \Lambda^{- \frac 1 2}P^T$.
   1. The whitened data is given by $X_\text{whitened} = VX_\text{centered}$
3. **Estimate the sources**. W/ ICA, your gaol is to find a lin trans $W = V^{-1}$ that unmixes the mixed signal & estimates the source signals. The independent sources $S_\text{estimated}$ are estimated using the equation $S_\text{esimated} = WX_\text{whitened}$.

### Factor analysis (FA)

- ~~***FA*** is a statistical technique used for "uncovering the underlying structure or latent factors that explain the observed correlatioons or patterns among a set of features." (My brother in Christ WHAT??)~~
- ***FA*** reduces dimensionality of data by "**identifying underlying latent factors** that explain observed features" (whatever tf THAT means).
  - Used to **study relationships between variables** & **reduce dimensionality** of data.
    - ^ frequently/commonly used in scientific fields such as psychology, social sciences, economics, & market research.
  - NOT used to identify causal relationships or test hypotheses about population means.
- ADVANTAGES:
  - Identifying "latent constructs"
  - Data interpretability
  - Dimensionality reduction
- DISADVANTAGES:
  - Assumption of linearity
  - Subjective interpretation
  - Sensitivity to data quality & sample size
- FA is a **two-step process**:
  - Step 1: ***Factor extraction***, which transforms features into "uncorrelated factors w/ corresponding loadings."
    - A ***factor loading*** $l_{ij}$ is a coefficient indicating the influence that factor $F_j$ has on feature $x_i$.
    - Factor loadings indicate strength & direction of relationship btwn common factors & features.
  - Step 2: ***Factor rotation***, which rotates "the axes" to make subsequent analysis easier.
    - Factors are rotated to maximize variance explained by fewer factors. This leads to **clearer interpretation of what each factor represents**.
- Variance in FA can be partitioned into common variance & unique variance.
  - A feature's ***common variance*** is the amt of **variance explained by all the factors**.
    - ***Communality*** is the term used to describe common variance btwn 0 & 1 that is equal to the sum of the squares of a feature's factor loadings.
  - ***Unique variance*** is the amount of **variance NOT shared** amongst a set of observed features.
    - May come from measurement error, or feature-specific reasons.

#### FA process

1. **Standardize the data**, since FA is sensitive to scale.
2. **Formulate a factor analysis** model. A factor model w/ $n$ factors can be represented as $\mathbb x = \Lambda F + \epsilon + \mu$, where&mdash;
    - $\mathbb x$ is a $p$-dimensional vec of feature vals.
    - $\mu$ is the $p$-dimensional vector of features' means.
    - $\Lambda$ is the $p \times n$ factor loading mtx, which represents relationship btwn common factors & observed features.
    - $F$ is an $n$-dimensional vector.
    - $\epsilon$ is a $p$-dimensional vec that represents variance coming from measurement error, noise, or feature-unique reasons.
3. **Extract the factors**. Common methods for factor extraction include PCA, maximum likelihood esimtation (MLE), and principal axis factoring (PAF).
4. **Rotate the factors**. Common methods of factor extraction include:
    - Varimax, which maximizes variance of the squared loadings.
    - Quartimax, which maximizes the sum of loadings raised to the 4th power.
    - Promax, which allows features to be correlated.
5. **Obtain factor loadings**. Most software packages that perform factor analysis include factor loadings in their output.