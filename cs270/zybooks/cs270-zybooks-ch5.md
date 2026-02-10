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