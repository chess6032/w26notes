IF YOU'RE VIEWING ON GITHUB: GitHub's LaTeX rendering in Markdown previews kinda sucks. I'd recommend downloading these and reading viewing them in VS Code&mdash;which I wrote them in. In particular, I uses `\mathbf` (bold) to indicate variables that are vectors, but GitHub doesn't appear to render that...RIP.

# 5.1: Linear Regression

Linear regression models a LINEAR RELATIONSHIP BTWN $p$ INPUT FEATURES: $x_1, x_2, \dots, x_p$, and a NUMERICAL OUTPUT feature: $y$.

Lin Reg is used for:

- DETERMINING THE RELATIONSHIP BTWN FEATURES, 
- FORECASTING, and
- PREDICTION.

## Simple Linear Regression

**Simple linear regression** models/predicts the output feature based on a LINEAR RELATIONSHIP w/ only ONE INPUT feature:

```math
\text{Simple linear regression (one input feature)}
```

```math
\hat y = w_0 + w_1x
```

- $w_0$ and $w_1$ are weights.
  - Mathematically, they're the y-intercept and the slope, respectively.

### Estimating weights

Given a bunch of data (i.e., data points $(x_k, y_k) \text{ for } 1 \le k \le p$), how do you come up with weights for the linear regression model (i.e., y-intercept & slope)?

**The desired model is one where predicted values ($\hat y_i$) are closed to observed data values ($y_i$)**. 

#### Estimating weights w/ LEAST SQUARES

Least squares is the MOST COMMON METHOD for estimating the weights of a lin reg model using available data.

The rationale for least squares starts with the **residual**, which is a way to calculate to the "closeness" of predicted to observed values. The residual just looks at the vertical distance btwn observed and predicted data: $e_i = y_i - \hat y_i$. 

Least squares selects weights $w_0$ and $w_1$ such that THE SUM OF SQUARED RESIDUALS (RSS) IS MINIMIZED:

```math
\text{RSS} = \sum_{i=1}^n (y_i - \hat y_i)^2 = \sum_{i=1}^n \big(y_i - (w_0 + w_1x_i)\big)^2
```

```math
\text{Least squares finds } w_0, w_1 \text{ such that RSS is minimized.}
```

## Multiple (Linear) Regression

More oft than not, multiple input features exist. 

```math
\text{Multiple linear regression (} p \text{ input features)}
```

```math
\hat y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p
```

- Where $w_k$ is the weight corresponding to feature $x_k$ for $1 \le k \le p$.
  - This can be thought like this: **$w_k$ represents the AVERAGE EFFECT on the OUTPUT feature for a one-unit increase in $x_k$**, if all other input feature values are fixed.

Instead of a line, multiple linear regression makes a HYPERPLANE.


Hehehe you could also write that in vector form:

```math
{\hat y} = \mathbf w \cdot \mathbf x
```

- Where $\mathbf w, \mathbf x \in \mathbb R^p$ represent your weights and input features, respectively.
  - $w_j$ corresponds to $x_j$ for $1 \le j \le p$.
  - $x_0$ is always equal to $1$.
  - One $\mathbf x$ is one data point.

(Note that $\mathbf u \cdot \mathbf v$ is often written as $\mathbf u^\text{T} \mathbf v$. This means "the transpose of $\mathbf u$ times $\mathbf v$", which is effectively the same as dot product&mdash;but thought of as matrix multiplication.)

### Least squares w/ multiple linear regression

Nothing actually changes.

In higher and higher dimensions, $y - \hat y$ is still just the "vertical" distance (i.e. **distance parallel to the $y$ axis**) btwn an observed data value and the hyperplane traced by $\hat y$. And the least squares algorithm still finds the values of $w_1, w_2, \dots , w_p$ such that produce the smallest possible sum of the squared residuals.

```math
\text{RSS} = \sum_{i=1}^n (y_i - \hat y_i)^2 = \sum_{i=1}^n (y_i - \mathbf w \cdot \mathbf x_i)^2
```

```math
\text{Least squares finds } w_0, w_1, w_2, \dots, w_p \text{ such that RSS is minimized} 
```

## Linear Regression in Python (5.1.1)

From what I gather, least squares regression for simple & linear models is done implicitly when you use `.fit()` in sklearn?

After fitting a model, the weights are stored as member variables:

- $w_0$ is found in the `.intercept_` attribute.
- $w_1, w_2, \dots, w_p$ is stored as an array in the `.coef_` attribute.

Additional details can be found in the [LinearRegression() documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

## Assumptions

The least squares linear regression method assumes the following assumptions are reasonably met:

- The relationship between output & input features is LINEAR and ADDITIVE.
  - LINEAR: Equal changes in $x$ cause equal changes in $\hat y$, regardless of what the values for $x$ are.
  - ADDITIVE: The relationship btwn input $x_i$ and the output is CONSTANT and does not depend on the value of any other input $x_j$.
- The RESIDUALS are INDEPENDENT.
- The RESIDUALS are NORMALLY DISTRIBUTED.
- The RESIDUALS have CONSTANT VARIANCE across the range of $x$ values.

^ I have NO clue what ts means.

Because of these assumptions, one constraint of least squares is setting the residuals' mean to $0$; other regression methods do not constrain the residuals' mean.

The impact of an unmet (or at least, not "reasonably met") assumption depends on which assumption it is.

## Advantages & Disadvantages of linear regression

Advantages:

- Computationally efficient and straightforward to interpret.
- Perform well when input & output features have a linear relationship and assumptions are reasonably met.

Disadvantages:

- Sensitive to outliers & exterme instnaces.
- Sensitive to multicollinearity&mdash;meaning multiple input features are correlated.
- Susceptible to noise & overfitting.

There do exist techniqeus to address these limitations. (Eg., regularization to prevent overfitting.)

# 5.2: Elastic Net Regression

## Regularization

- **Regularization** refers to methods that CONTROL THE COMPLEXITY of an ML model by adding a regularization term which SHRINKS WEIGHTS TOWARD $0$.
  - Note however that regularization ONLY AFFECTS WEIGHTS CORRESPONDING TO INPUTS.

Often times, regularization methods will have a hyperparameter&mdash;often noted as $\alpha$ or $\lambda$&mdash;representing the amount of regularization. Hence, as this hyperparam increases, weights shrink to $0$:

```math
\text{Regularization shrinks weights to zero}
```

```math
\lim_{\alpha \rightarrow \infty}w_j = 0
```

```math
\forall \space 1 \le j \le p
```

Different regularization methods differ in their LOSS FUNCTION.

- A **loss function** quantifies the DIFFERENCE between a MODEL'S PREDICTIONS and OBSERVED VALUES.
  - Models are fitted by solving for weights that minimize the loss function.

e.g., the loss function for least squares is the sum of squared residuals ($\text{RSS}$).

### Ridge Regression (i.e. L2 Regularization)

Ridge regression starts with $\text{RSS}$, but adds an **L2 Norm Regularization** term to the loss function. 

The term it adds is thus:

```math
\text{L2 Norm Regularization term}
```

```math
\alpha || \mathbf w || _2 ^2 = \alpha \sum_{j=1}^p w_j^2
```

- Where $\alpha \ge 0$ is a hyperparameter, and $\mathbf w \in \mathbb R^p$ represents your weights associated with input features&mdash;$w_j$ corresponds to feature $x_j$.
  - **Increasing the value of $\alpha$ increases the amount of regularization.** (Remember: With more regularization, estimated weights shrink toward zero.)
- $|| \space \space ||_2 ^2$ is the mathematical notation for the "L2 norm" of a vector, meaning the sum of each component squared.

Hence, the final loss function is thus:

```math
\text{Ridge regression}
```

```math
\text{Loss}=\sum_{i=1}^n (y_i - \hat y_i)^2 + \alpha \sum_{j=1}^p w_j^2
```

```math
\text{During fitting, selects weights } w_0, w_1, w_2, \dots, w_p \text{ such that Loss is minimized}
```

### LASSO Regression (i.e. L1 Regularization)

("LASSO" stands for "Least Absolute Shrinkage and Selection Operator).

LASSO adds this term to the loss function:

```math
\text{L1 Norm Regulrization term}
```

```math
\alpha || \mathbf w || _1 = \alpha \sum_{j=1}^p |w_j|
```

- Where $\alpha \ge 0$ is a hyperparam, and $\mathbf w \in \mathbb R^p$ weights associated w/ input features.
- $||\space\space||_1$ is the mathematical notation for "L1 norm" of a vector, meaning the sum of the absolute values of each component.

Thus, the final loss function looks like this:

```math
\text{LASSO regression}
```
```math
\text{Loss} = \sum_{i=1}^n (y_i - \hat y_i)^2 + \alpha \sum_{j=1}^p |w_j|
```
```math
\text{During fitting, selects weights } w_0, w_1, w_2, \dots, w_p \text{ such that Loss is minimized}
```

Unlike Ridge regression, LASSO has the power to reduce weights COMPLETELY TO ZERO. Hence, LASSO can be used for FEATURE SELECTION: if a weight is shrunk to zero, its corresponding input feature must not be important. (Because if $w_k$ is shrunk to zero, then the term $w_k x_k$ cancels out to zero&mdash;in which case $x_k$ isn't even contributing to $\hat y$ anymore!)

^ More LASSO regularization means more weights reduce to zero. With a high enough $\alpha$ value, *all* weights corresponding to input features ($w_1$ to $w_p$) reduce to zero&mdash;leaving you with an intercept only model (the only term left is $w_0$).


### Elastic Net Regression

Elastic Net regression COMBINES RIDGE & LASSO to (ideally) get the best of both worlds.

Here's the term it adds:

```math
\text{Elastic Net regularization term}
```

```math
\alpha \Big( \lambda ||\mathbf w||_1 + (1-\lambda) ||\mathbf w||_2 ^2 \Big)
```

- $\alpha \ge 0$ controls the **regularization strength overall**.
- $0 \le \lambda \le 1$ is the **weight applied to the L1 norm**.
  - HIGHER VALUES favor LASSO regression. When $\lambda = 1$, it becomes equivalent to LASSO regression.
  - LOWER VALUES favor RIDGE regression. When $\lambda = 0$, elastic net becomes equivalent to ridge regression.
  - Hence, **tuning $\lambda$ determines the best ratio between L1 and L2 regularization.**

> Note: Sometimes, Elastic Net's regularization term is written as $\alpha_1 ||\mathbf w||_1 + \alpha_2||\mathbf w||_2^2$. So instead of having one hyperparam represent the strength of regularization overall and another to represen the ratio of L1 to L2, L1 and L2 each just get their own independent hyperparameters.

Hence, the final loss function is thus:

```math
\text{Elastic Net Regression}
```

```math
\text{Loss}=\sum_{i=1}^n (y_i - \hat y_i)^2 + \alpha \bigg(\lambda \sum_{j=1}^p |w_j| + (\lambda - 1)\sum_{j=1}^p w_j^2 \bigg)
```

```math
\alpha \ge 0\text{ controls strength of regularization}, 0 \le \lambda \le 1\text{ controls the ratio between L1 and L2 regularization}
```


So should you lean more into L1 or L2? I played around with the interactive thing in 5.2.7 and determined thus:

<!-- - **L1: Lower error variance, higher prediction variance.** -->
<!-- - **L2: Higher error variance, lower prediction variance.** -->

- **L1: HIGHER PREDICTION variance, LOWER ERROR variance.**
- **L2: LOWER PREDICTION variance, HIGHER ERROR variance.**

### Some notes on regularization

The following are true for Ridge, LASSO, AND Elastic Net regression.

- Increasing the value of $\alpha$ increases the amount of regularization.
  - **Increasing regularization decreases prediction variance.** (good!)
    - (Predictions become more stable/consistent.)
  - **Increasing regularization increases bias.** (bad)
    - (Predictions may be systematically off b/c the model is too constrained.)
- Increasing the value of $\alpha$ increases the minimum value of the loss function.
  - Because you're increasing the penalty for complexity, I guess?
  - I think it's for this reason that **error variance increases when you increase** $\alpha$.
- As the value of $\alpha$ increases, the estimated weight for $w_0$ stays the same, because **regularization only affects weights associated with input features**.
- Setting $\alpha = 0$ is equivalent to least squares regression.
  - Because $\alpha$ is the coefficient to the regularization term. Setting it to zero cancels out that term entirely, leaving you only with $\text{RSS}$.

## Regularization in sklearn

- Elastic Net Reg is implemented in sklearn via `ElasticNet()` from `sklearn.linear_model`.
- INPUT FEATURES SHOULD BE SCALED (using `StandardScaler()`) BEFORE FITTING a model using a regularization method.

### `ElasticNet()` Parameters

| Parameter  | Default | Description |  
| ---------  | ------- | ----------- |  
| `alpha`    | `1.0`   | Controls the strength of regularization. |  
| `l1_ratio` | `0.5`   | Specifies the weight applied to the L1 regularization term; hence, it controls the ratio btwn L1 & L2 regularization. |  
| `max_iter` | `1000`  | Sets the max number of iterations. (The Zybooks didn't say what we're iterating through or for or why...) |  

In the equations for Elastic Net Regression above, `alpha` is $\alpha$ and `l1_ratio` is $\lambda$.

## Model tuning

- **Model tuning** is the process of selecting the best hyperparam val(s) for a model. 
  - e.g., $\alpha$ in ridge or LASSO regression.

## Advantages & Disadvantages of Regularization

Advantages:

- LASSO and elastic net weights can be estimated at 0, which results in a simpler model. Simpler models can be EASIER TO INTERPRET and are MORE COMPUTATIONALLY EFFICIENT.
- **Regularization discourages large weight estimates, thus PREVENTING OVERFITTING.**

Disadvantages:

- Ridge regression only shrinks weights close to 0, but never completely to 0. So its computational expenditure is similar to linear regression.
- **Regularization REDUCES PREDICTION VARIANCE but at the cost of INTRODUCING BIAS.**
  - **Bias-variance tradeoff.**
  - Thus, the selection of hyperparams is important.
    - Ideally, for a small increase in bias you can get a large decrease in prediction variance.
  - This tradeoff does not exist with least squares because **least squares constrains the mean of the residuals to 0, providing unbiased estimates of the weights.**

## Misc

- When an input feature is standardized, $w_0$ (the y-intercept) is the AVERAGE OF THE OUTPUT FEATURE.
  - idrk what "standardized" means in this context...
  - (5.2.3)
- For a simple linear model, the loss function for ridge regression simplifies to $\text{RSS} + \alpha w_1^2$. Neat!

# 5.3: kNN for Regression

$k$-nearest neighbors regression predicts the value of a numeric output feature based on the avg output for other instances w/ the most similar (i.e. nearest) input features.

So it's just KNN classification, except instead of using similar neighbors to predict a class, you're predicting a number. **The average value of the output feature for the $k$ nearest instances becomes the prediction.**

- Linear regression assumes input and output features are related.
- KNN regression assumes instances close to each other are similar.

## KNN Regression Algorithm

- LET $\mathbf x_i = [x_{1i}, x_{2i}, \dots, x_{pi}]$ denote the $p$ input features from instnace $i$.
- LET $y_i$ denote the output feature.
- LET $n$ denote the number of instances in a dataset.

The output value for a new instance $\mathbf x^*$ is determing in KNN reg by IDENTIFYING THE $k$ NEAREST INSTANCES TO $\mathbf x^*$, and ASSIGNING THE AVG OUTPUT VAL of these instances as the prediction.

A step-by-step algorithm is describe in 5.3.1:

1. Select an integer val for $k$.
2. Calculate the distance btwn $\mathbf x^*$ and all other $\mathbf x_i \space \forall \space 1 \le i \le n$.
3. Sort the distance from smallest to largest, and identify the $k$ smallest distances. Let $N$ be the set of the indeces ($i$) of the $k$ nearest neighbors to $\mathbf x^*$.
4. Calculate the average of the output features for the $k$ nearest neighbors: $\frac 1 k \sum_ {i \in N} y_i$
5. The average output for the nearest neighbors is the prediction for instance $\mathbf x^*$.

I guess the big thing is this:


```math
\text{LET } N \text{ be set of the indeces of the } k \text{ nearest neighbors to } \mathbf x^*
```

```math
y^* = \frac 1 k \sum_{i \in N} y_i
```

```math
(\text{The output of } \mathbf x^* \text{ is the average of the } k \text{ nearest neighbors' outputs})
```

## Selecting the nearest neighbors

KNN has TWO HYPERPARAMS:

1. The number of neighbors ($k$).
2. The metric for measuring distance.

&nbsp;

- Setting $k$ too SMALL may result in OVER-fitting.
- Setting $k$ too LARGE may result in UNDER-fitting.

Meanwhile, depending on how distance is defined, the nearest neighbors may change. Three common distance measures for kNN regression are EUCLIDEAN, MANHATTAN, and MINKOWSKI distance (the same ones for KNN classification). 

Here is each distance formula for determing the distance btwn two instances, $a$ and $b$, with $p$ input features (5.3.1):

&nbsp;

```math
\text{Manhattan distance}
```
```math
d_M(a, b) = \sum_{i=1}^p |x_{ia} - x_{ib}|
```

&nbsp;

```math
\text{Euclidean distance}
```
```math
d_E(a, b) = \sqrt{\sum_{i=1}^p (x_{ia} - x_{ib})^2}
```

&nbsp;

```math
\text{Minkowski distance}
```
```math
d_m(a, b) = \Big(\sum_{i=1}^p |x_{ia} - x_{ib}|^m\Big)^{1/m}
```

If you look closely, you'll notice that Manhattan and Euclidean distance are really just Minkowski distance with $m = 1$ and $m = 2$, respectively.

## kNN Regression in sklearn

- kNN is implemented via `KNeighborsRegressor()` from `sklearn.neighbors`.
- `model.kneighbors(X)` method returns the indeces and distances of the $k$ nearest neighbors for `X`.

### `KNeighborsRegressor()` parameters

| Parameter     | Default         | Description |  
| ------------- | --------------- | ----------- |  
| `n_neighbors` | `5`             | Sets the number of neighbors ($k$) |  
| `metric`      | `"minkowski"`   | Distance measure for calculating distance btwn points. (See [this documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics) for a full list of allowable metrics.) |  
| `p`           | `2` (Euclidean) | Power parameter $m$ for Minkowski distance. Setting `p=1` or `p=2` results in Manhattan or Euclidean distance, respectively. |  

More params & details can be found in the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).

## Advantages & Disadvantages

Advantages:

- kNN regression is a flexible model and DOES NOT REQUIRE A SPECIFIC RELATIONSHIP BTWN INPUT & OUTPUT features.
  - Hence, the model WORKS WELL FOR COMPLEX OR NON-LINEAR RELATIONSHIPS.
- With a large enough $k$, the model is NOT sensitive to noise or outliers.

Disadvantages:

- kNN regression is SENSITIVE to the NUMBER OF INPUT FEATURES and the NUMBER OF INSTANCES.
  - When the number of input features becomes large, the model can end up having sparse areas. Hence, the nearest neighbors for an instance $\mathbf x^*$ may not actually be similar to $\mathbf x^*$, resulting in poor predictions.
  - Because the kNN algorithm calculates the distance from each instance, the algorithm is COMPUTATIONALLY EXPENSIVE w/ a large number of instances.

When a large number of input featuers is used to fit a KNN regression model, the data should also include a large number of instances. (Avoids sparse areas in model, ensuring that neighbors *are* actually similar to each other.)

# 5.4: Loss functions for regression

- A loss function for regression measures how close a model's predictions are to actual values.
  - The output of a loss function is always a **non-negative** value. (Meaning it's $\ge 0$)
  - Smaller loss values indicates the model fits the data better&mdash;i.e. better predictions.
- Regression models are fitted by selecting the weights that minimize a specified loss function.

Common loss functions for regression include ABSOLUTE LOSS, SQUARED LOSS, HUBER LOSS, and QUANTILE LOSS.

## Absolute Loss

- Absolute loss is **just the residual**&mdash;also called **"prediction error"**&mdash;which is just the (absolute value of the) distance of the predicted value ($\hat y$) from the observed (actual) value ($y$).

```math
\text{Absolute Loss}
```
```math
L_{abs} (y_i, \hat y_i) = |y_i - \hat y_i|
```

Absolute loss is also called **"L1 norm loss"**.

### Mean Absolute Error (MAE)

- The mean absolute error (MAE) is the AVERAGE ABSOLUTE LOSS FOR ALL INSTANCES:

```math
\text{Mean Absolute Error}
```

<!-- ```math
\text{MAE} = \frac 1 n \sum_{i=1}^n L_{abs}(y_i, \hat y_i) = \frac 1 n \sum_{i=1}^n |y_i - \hat y_i|
``` -->

```math
\text{MAE} = \frac 1 n \sum_{i=1}^n |y_i - \hat y_i|
```

<!-- ```math
\text{The average absolute loss for all instances}
``` -->

## Squared Loss

Squares loss defines loss as the SQUARED PREDICTION ERROR (i.e. SQUARED RESIDUAL), which **places more emphasis on predictions far from observed values**, compared to absolute loss.

Squared loss has NICE MATHEMATICAL PROPERTIES and is A COMMON CHOICE for fitting regression models. E.g., the **least squares method uses squared loss**.

- The squared loss for an instance is the SQUARED DIFFERENCE BTWN OBSERVED ($y_i$) AND PREDICTED ($\hat y_i$) VALUES.

```math
\text{Squared Loss}
```

```math
L_{sq}(y_i, \hat y_i) = (y_i - \hat y_i)^2
```

The squared loss is also called the **"L2 norm loss"**.

### Mean Squared Error (MSE)

- The mean squared error (MSE) is the AVERAGE SQUARED LOSS FOR ALL INSTANCES.

```math
\text{Mean Squared Error}
```

```math
\text{MSE} = \frac 1 n \sum_{i=1}^n (y_i - \hat y_i)^2
```

## Huber Loss

Huber Loss is a COMBINATION OF ABSOLUTE & SQUARED LOSS.

- Huber loss is similar to SQUARED LOSS when PREDICTION ERRORS ARE SMALL. (Higher values of $\epsilon$.)
  - This provides the mathematical benefit of being differentiable at zero, which is beneficial for efficient model fitting.
- Huber loss is similar to ABSOLUTE LOSS when PREDICTION ERRORS ARE LARGE. (Lower values of $\epsilon$.)
  - This way, Huber loss is LESS SENSITIVE TO OUTLIERS.

Huber loss is a piecewise function:

```math
\text{Huber Loss}
```

```math
L_{Huber}(y_i, \hat y_i) = \begin{cases}
  \frac 1 2 (y_i - \hat y_i)^2 & \text{if } |y_i - \hat y_i| \le \epsilon \\
  \epsilon(|y_i - \hat y_i| - \frac 1 2 \epsilon) & \text{if } | y_i - \hat y_i| > \epsilon
\end{cases}
```

- Where $\epsilon > 0$ is a hyperparam that **sets the point where the loss transitions from quadratic (squared loss) to linear (absolute loss)**.
  - A LARGER value of $\epsilon$ makes Huber loss more similar to SQUARED loss. (And consequently, PREDICTION ERRORS are SMALLER.)
  - A SMALLER value of $\epsilon$ makes Huber loss more similar to ABSOLUTE loss. (And consequently, PREDICTION ERRORS are LARGER.)

When graphed, the Huber loss function is just a normal quadratic for $-\epsilon \le y-\hat y \le \epsilon$. Then, at the edges of that interval, the function stops changing and just shoots out in a straight line. (It's kinda cool looking&mdash;I like it.)

![ts kinda fire](cs270-zb-ch5.4.7-huber-loss.png)

I guess this is why $\epsilon$ is so weirdly thrown into the linear part of the Huber loss function&mdash;it's to line up the absolute loss function with the squared loss function, such that they have the same value & derivative at $y - \hat y = \pm \epsilon$.

## Quantile loss

In the quantile loss function, **whether an innstance is under- or over-predicted affects the instance's loss**.

The quantile loss function is used for fitting **quantile regression** models, which model/predict a SPECIFIC QUANTILE of the numeric output feature based on the input features. (e.g., a quantile regression model predicts the 10th quantile of insulin level based on glucose level. Another quantile regression model predicts the 90th quantile of insulin level based on glucose level.)

The quantile being predicted affects the loss of under- and over-predictions. (See hyperparameter for equation below.)

The quantile loss for an instance is:

```math
\text{Quantile Loss}
```

```math
L_{quant} (y_i, \hat y_i) = q \max(0, y_i - \hat y_i) + (1 - q) \max(0, -(y_i - \hat y_i))
```

- Where $0 \lt q \lt 1$ is the **quantile being predicted**.
  - For predicting SMALL quantiles, OVERPREDICTED instances have higher loss.
  - For predicting LARGE quantiles, UNDERPREDICTED instances have higher loss.
  - For quantiles close to 0.5, $q$ and $1-q$ are similar, so the quantile loss is influenced more by the MAGNITUDE OF THE PREDICTION ERROR and less by whether an instance is over/under predicted.

![ts is NOT tuff](cs270-zb-ch5.4.9-quantile-loss.png)

- OVER-predicted: $y_i - \hat y_i < 0$
- UNDER-predicted: $y_i - \hat y_i > 0$

The overall quantile loss is summarized by the mean quantile loss for all instances.

## Loss functions in sklearn

Sklearn's loss functions come from the `sklearn.metric` module:

- absolute loss: `mean_absolute_error()`.
- squared loss: `mean_squared_error()`.
  - To initialize a linear regression model that uses squared loss, use `LinearRegression()`
- quantile loss: `mean_pinball_loss()`.
  - (wtf.)


Each of these functions take an array of OBSERVED output values and an array of PREDICTED output values. Additional details can be found in the [regression metrics documentation](https://scikit-learn.org/stable/modules/classes.html#regression-metrics).

For fitting w/ Huber loss in sklearn, use `HuberRegressor()`. Parameters and additional details can be found in the [HuberRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html).

(Thanks for nothing Zybooks.)
