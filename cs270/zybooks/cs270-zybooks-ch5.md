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

### Least squares w/ multiple linear regression

