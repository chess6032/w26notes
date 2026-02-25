# 6.1: Decision trees for classification

## Trees

A tree is a hierarchical structure of nodes & edges with **no loops**.

- A root node is a node with no parent.
- A leaf is a node w/ no children.
- The **depth** of a node is the (minimum) **number of edges it takes to reach that node from the tree's root**.

## Decision Trees

- A ***decision tree*** is a tree where each node has **a QUESTION that determines WHICH CHILD to descend to**.
- In decision trees, each LEAF represents a FINAL DECISION.

## Decision Tree Classifiers (DTC)

Decision tree classifiers are decision trees built for classification.

- **Each LEAF is a CLASS** from the OUTPUT feature.
- At each **NODE**, the decision is based on EITHER:
  - a) an **INEQUALITY of NUMERICAL features**.
    - e.g. acidity &le; 5.5
  - b) an **EQUALITY for CATEGORICAL features**.
    - e.g. roast == 'dark'

When training a DTC model, we keep track of the number of instances of each class that lie at each leaf. (And maybe at each decision, too?) This data is used for calculating the confidence of predictions.

### Graphically representing decision trees

6.1.3: 

![hey it kinda looks like the fibonacci sequence](images/cs270-zb-6.1.3-decision-tree-classifier-graph.png)

- Each DECISION RULE divides the feature space w/ HYERPLANES that are PERPENDICULAR to the DECISION FEATURE'S AXIS.
  - Each REGION resulting from these divisions MATCHES A LEAF.
- The PREDICTED CLASS for each leaf is the MOST COMMON CLASS IN THE MATCHING REGION

<!-- ### Deciding on decisions

How do you determine what to put for each node's decision?

I am fully talking out my ahh here but here's what I'm picking up:

- Each decision should do the best it can to separate classes.
  - e.g., if you're trying to determine whether your Valentine's Day gift is good, adding a decision splitting btwn `Dark Chocolate` and `Flowers` would be pretty useless, since both are beloved by women.
- The specificity of each decision should increase as you get further from the root.
  - I.e., earlier nodes should split the dataset into big chunks (ideally close to half-and-half, I'd assume).
  - This ensures you have to perform fewer checks, meaning you'll reach a leaf faster.
    - I.e., it ensures your decision tree is more balanced.


### Complexity vs. Accuracy

- A leaf that captures INSTANCES of ALL THE SAME CLASS is called a **pure leaf**.
- A 100% ACCURATE decision tree would have ONLY PURE LEAVES...but it would also be VERY COMPLEX, increasing computation time and stuff. 
  - So ig there's like a balance between how accurate your dec tree is and how complex it is. -->

## Decision Tree Classifiers in sklearn

Sklearn implements DTCs w/ **`DecisionTreeClassifier()` from `sklearn.tree`**.

The syntax is basically the same as all models we've used thus far:

- `.fit(X, y)` and `.predict(X)` train the model and make predictions, respectively.
- `.predict_proba(X)` predicts class probabilities (i.e. the probability that a classification is correct).
  - For DTCs, this is the proportion of training samples of that class in the leaf node.
- Class labels can be accessed with the `.classes_` attribute.

## Measures of fit (i.e. evaluating a DTC...I think?)

A DTC's measures of fit are based on **how well the node's decisions SEPARATE CLASSES.**

- The ***purity*** of a node is the PROPORTION of the node's INSTANCES that are in said node's MOST FREQUENT CLASS.
  - A node has $\text{purity} = 1$ if ALL node's instances are in the SAME class.
    - Such nodes are called **"pure"**. (Or at least leaf nodes are.)
- The ***impurity*** of a node describes how much the node fails to be pure.

In practice, **REDUCING IMPURITY is more effective when training decision trees.**

### Measuring impurity

Two commonly used impurity measures for a node are GINI IMPURITY and ENTROPY.

For the equations below, note the following:

- The proportion of instances of class $k$ at node $i$ is represented by $p_{ki}$.
- A tree's impurity is the WEIGHTED AVG of EACH LEAF'S IMPURITY measure: $\frac {\sum_{i} n_i \text{impurity}_i} {\sum_{i} n_i}$
  - $n_i$ is the NUMBER OF INSTANCES in leaf $i$.
  - $\text{impurity}_i$ is the impurity of leaf $i$.
- For both Gini impurity & entropy, a node's impurity reaches a MAXIMUM when the CLASS PROPORTIONS ARE EQUAL.
  - i.e. $p_{ki} = \frac 1 k \forall k$

#### Gini impurity


```math
\text{Gini Impurity}
```
```math
\sum_k p_{ki} (1 - p_{ki})
```

- Gini impurity is QUADRATIC IN NATURE.
- To use Gini impurity in sklearn, you'd put `criterion="gini"` into the parameters of `DecisionTreeClassifier()`.
  - Actually you don't technically have to&mdash;**Gini impurity is the DEFAULT for sklearn**.

#### Entropy (i.e. log loss)

```math
\text{Entropy}
```
```math
-\sum_k p_{ki} \ln(p_{ki})
```

- The minus sign ensures the impurity stays a POSITIVE NUMBER.
  -  bc $\ln(x)$ is negative when $x<1$, and all probabilities lie between $0$ and $1$.
- DIFFERENT LOG BASES CAN BE USED.
   - This impacts ONLY the SCALE of the impurity measure, NOT the RELATIVE COMPARISON of node impurities.
- To use entropy in sklearn, you'd put `criterion="entropy"` into the parameters of `DecisionTreeClassifier()`.

#### Note

- **The split that minimizes entropy may not minimize Gini impurity, and vise versa.**

## Preventing overfitting

A SUPER DEEP DTC indicates OVERFITTING to the training data. (e.g., a tree so deap that each leaf contains one instance.)

### Early stopping

- ***Early stopping*** is the use of a NODE'S STATS to stop early, PREVENTING FURTHER GROWTH below that node, thus REDUCING OVERFITTING.

#### Adjusting early stopping conditions for a `DecisionTreeClassifier()`

For sklearn's `DecisionTreeClassifier()`, you add early stopping behavior via PARAMETERS.

| Param               | Default | Description | Notes |  
| -----               | ------- | ----------- | ----- |  
| `max_depth`         | `None`  | Maximum depth of leaf nodes in the trained tree. | Leaves may have smaller depth if other early stopping conditions are met. |  
| `min_samples_split` | 2       | Minimum instances a node must have to be considered for splitting. Nodes w/ fewer instances than `min_samples_split` are left as leaves. | `min_samples_split` may be set to a number of instances (INT), OR a proportion of the number training instances (FLOAT). |  
| `min_samples_leaf`  | 1       | Minimum instances that must be in each leaf. A split that would lead to a leaf further down w/ fewer than `min_samples_leaf` is not considered. | `min_samples_leaf` may be set to a number of instances (INT), OR a proportiong of the number of training instances (FLOAT). |  
| `max_leaf_nodes`    | `None`  | Maximum number of leaves in the resulting tree. Splits that have the greatest relative reduction in impurity are added first. | *(I don't understand the description for this one...)* |  

### Cost complexity pruning

Another method to prevent overfitting is to PRUNE the resulting tree.

- ***Pruning*** is the process of TRUNCATING A TREE by turning a DEICSION NODE INTO A LEAF.
  - Which decision nodes are pruned is based on the **COMPLEXITY COST** of the TREE BELOW EACH NODE.

**Cost complexity balances a tree's IMPURITY measure against THE NUMBER OF LEAVES (i.e COMPLEXITY) in the tree**. Here is its formula:

```math
\text{Complexity Cost}
```
```math
R_\alpha(T) = R(T) + \alpha |\~T|
```

- $T$ is the tree.
- $R(T)$ is the impurity measure of $T$.
- $|\~T|$ is the NUMBER OF LEAVES in $T$.
- $\alpha$ is a PARAMETER that CONTROLS THE BALANCE btwn IMPURTY & COMPLEXITY.

For each node $t$ and the subtree descending from it, $T_t$, the node's cost as a leaf, $R_\alpha(t)$, is compared to the cost of the node's subtree, $R_\alpha(T_t)$. IF $R_\alpha(t) < R_\alpha(T_t)$, ALL nodes below $t$ are PRUNED.

```math
R_\alpha(t) < R_\alpha(T_t) \Rightarrow \text{prune } T_t
```

(Note: $|\~t| = 1$. When calculating $R_\alpha(t)$ for a node $t$, you look at the node in isolation&mdash;NOT as the root of its subtree.)

#### Adjusting cost complexity of a `DecisionTreeClassifier()`

- To enable cost complexity pruning, set the `ccp_alpha` param of `DecisionTreeClassifier()` to a POSITIVE VALUE.
- By default, NO pruning is done.

## Advantages & Disadvantages of DTCs

ADVANTAGES

- Small trees are interpretable by people. (They're just flowcharts.)
- Trees can be translated into yes/no rules.
- Data does NOT need to be standardized, bc only one feature is considered at a time.
  - Furthermore, it's therefore not too hard to adapt decision trees to use missing values.
- Important features for separating the classes appear at the top of the tree. (How intuitive!)

DISADVANTAGES

- Large trees are less interpretable, since they're harder to view and there's more to consider.
- Splits occur only on one feature.
  - If data has correlated features, prepare for deep trees.
    - (You'll end up w/ a stair-step pattern.)

# 6.2: Decision Tree algorithms

## Divide-and-conquer

"Divide-and-conquer", in this context, refers to **top-down, recursive algorithms on trees**.

- In divide-and-conquer algorithms for training decision trees, we **start at the root node and SPLIT THE DATASET on the INPUT FEATURE that IMPROVES PREDICTION THE MOST**.
  - The splitting process is then repeated on each child&mdash;unless no further improvement in prediction can be made.
  - Ig "improves prediction the most" just means the split that results in the lowest impurity? Idk.

### Basic divide-and-conquer Algorithm

Uhhh ig it might look smth like this:

$\text{Grow}(T):$  
$\text{for every possible split } s:$  
$\text{compare } s \text{ to best performing split so far, } s_{best} \text{. update if better.}$  
$\text{end for}$  
$\text{split } T \text{ by } s_{best} \text{ into } T_1, T_2, \dots, T_n$  
$\text{for } T_i \text{ in } 1 \le i \le n:$  
$\text{Grow}(T_i)$  
$\text{end for}$  
$\text{end Grow}$  

## Iterative Dichotomiser 3 (ID3)

ID3 is a divide-and-conquer algorithm for creating decision tree classifiers.

- At each node, ID3 chooses the split that results in the **lowest ENTROPY**.
- ID3 grows a complete tree where at every leaf you know EITHER:
  - the **leaf is pure**, 
  - OR **every input feature has been used** in the leaf's decision path.
- All **features must be converted to CATEGORICAL values**.
  - Hence, **ID3 only works for CLASSIFICATION**.
  - (ID3 is old (1986). It is designed to work w/in the computing constraints of its time.)

## C5.0

### ID3 &rightarrow; C4.5

C4.5 is a DT algorithm by the same dev of ID3.

The textbook alleges that C4.5 was one of the most successful ML algorithms in the lat 90's and early 2000s. (Some official people in suits would even put it in [the top ten!](http://www.cs.umd.edu/~samir/498/10Algorithms-08.pdf))

C4.5 improvded on ID3 by:

- allowing use of NUMERICAL FEATURES.
- using MISSING DATA.
- assigning COSTS to features.
  - A feature's cost is a PENALTY that makes a feature LESS LIKELY TO BE SELECTED for a decisionnode.
- PRUNING TREES after creation.
- converting the final DT to a RULESET: A set of IF-ELSE STATEMENTS that predict an instance.
  - "A ruleset's statements can be rearranged in a method similar to pruning but with different outcomes." (Huh?)

### C4.5 &rightarrow; C5.0

C5.0 added:

- Support for weighting misclassifications (false pos vs. false negs) differently.
- BOOSTING: fits multiple DTs in sequence. Misclassifications in early trees are weighted more heavily when training later trees. 
- WINNOWING: estimates the increase in error rate if a feature is removed. Then, before growing, features that have small effects on error rate are removed. 

### C5.0 in sklearn

SIKE! Sklearn doesn't have an implementation of C5.0, but it is available through ML packages in R.

## Classification and Regression Trees (CART)

CART is a divide-and-conquer algorithm that builts DTs for both **CLASSIFICATION AND REGRESSION** probs using both **NUMERICAL AND CATEGORICAL FEATURES.**

It was first published in 1984. (JORJOR WELL REFERENCE!)

- CART builds a **BINARY TREE**. 
  - (At every node, the dataset is split into TWO groups.)
- CART may use **ANY METRIC** for determing the best split at each node.
  - Classification: Gini impurity, entropy, log-loss, etc.
  - Regression: MSE, MAE, etc.
- Predictions:
  - Classification: **Most common output** for each leaf's training instances.
  - Regression: **mean (or median) of each leaf's instances' outputs**.
- CART **prunes after growing**, using cost-complexity pruning.

### CART in sklearn

**Sklearn implements DTs via CART**, since it can do both classf. and regrsn. However, in either case, **CATEGORICAL input features must be CONVERTED TO DUMMY VARIABLES before use.**

## Comparison of algorithms

| Algorithm | Year released | Model type                   | Input feature type       | Pruning supported?| Rulesets as output? | Available in sklearn? |  
| --------- | ------------- | ---------------------------- | ------------------------ | ----------------- | ------------------- | --------------------- |  
| ID3       | 1983          | Classification               | Categorical              | ✕                 | ✕                  | ✕                     |  
| C5.0      | 2011          | Classification               | Categorical OR numerical | ✓                 | ✓                  | ✕                     |  
| CART      | 1984          | Classification OR regression | Categorical OR numerical | ✓                 | ✕                  | ✓                     |  

All algorithms discussed in this chapter are divide-and-conquer algorithms. They are also all **greedy algorithms**, meaning they solve a problem in multiple steps, taking the best option at each step.

Greedy algorithms are **not guaranteed to find the overall optimal solution**. They are used for find **SUBOPTIMAL SOLUTIONS for problems that are TOO EXPENSIVE to find the optimal solution.**

# 6.3: Decision trees for regression

## Modifying DTs for use w/ regression

When using DTs for regression:

- Decisions split w/ inequalities of input features.
  - As is always the case, each split splits along only one input feature. So at each decision node, there's a SINGLE INEQUALITY that splits the dataset.
- Once at a leaf, its predicted value is set to the OUTPUT FEATURE'S MEAN/MEDIAN for the instances in that leaf's region.

## Error Estimates

- During training, different error estimates may be used.
- In sklearn, the error estimate is specified by the `criterion` parameter when instantiating a `DecisionTreeRegressor`.
  - The default is `"squared_error"` (MSE)

For the formulas below:

- $R_i$ is the region corresponding to an individual node.
  - (To evaluate an ENTIRE TREE, the WEIGHTED AVERAGE OF ALL LEAVES is calculated, where each leaf's NUMBER OF INSTANCES provides its weight.)
- $n_i$ is the NUMBER OF TRAINING INSTANCES in region $i$.

### Mean squared error (MSE)

```math
\text{MSE}
```
```math
\frac 1 {n_i} \sum_{y\in R_i} (y - \bar y)^2
```
```math
\texttt{"squared\_error"}
```

- MSE is the **DEFAULT CRITERION for DTs in sklearn**.
- **Good GENERAL USE** estimate:
  - does not require expensive computations.
  - leads to efficient convergence (under most conditions.)
    - (no idea what ts means.)

### MSE w/ Friedman's Improvement

```math
\text{MSE with Friedman's improvement}
```
```math
\frac{n_{below} \times n_{above}}{n_{below} + n_{above}} (\bar y_{below} - \bar y_{above})^2
```
```math
\texttt{"friedman\_mse"}
```

- Not *technically* an error measurement, since it doesn't use instance outputs.
- Larger improvements indicates the threshold separates two different outputs well. (Huh?)
- **Useful in cases when CLASSES ARE IMBALANCED**:
  - Favors an EQUAL NUMBER OF INSTANCES IN EACH CHILD
  - provides some NUMERICAL STABILITY.

### Mean Absolute Error (MAE)

```math
\text{MAE}
```
```math
\frac 1 {n_i} \sum _{y \in R_i} |y-m_i|
```
```math
\texttt{"absolute\_error"}
```

- ($m_i$ is MEDIAN)
- Use is appropriate when **OUTPUT FEATURE IS SKEWED**:
  - Uses MEDIAN AS PREDICTION INSTEAD OF MEAN.
- HOWEVER, finding median requires MORE COMPUTATIONS than mean, so **trees will TAKE LONGER TO TRAIN using MAE** than using MSE.

### Half Poisson deviance

```math
\text{Half Poisson deviance}
```
```math
\frac 1 {n_i} \sum_{y\in R_i} \Big(y\ln \frac y {\bar y_i} - y + \bar y_i \Big)
```
```math
\texttt{"poisson"}
```

- Best used when OUTPUT FEATURE IS A COUNT/FREQUENCY.
- Log algorithm makes it EXPENSIVE TO CALCULATE, so trees trained w/ half Poisson deviance TAKE LONGER TO TRAIN than those w/ MSE.


## DT Regression in Sklearn

DT reg. is implemented in sklearn using `DecisionTreeRegressor()` from `sklearn.tree`. ([Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html))

### Parameters

I'm *pretty* sure all the parameters for DT classifier also apply to the DT regressor. Here's the ones from the Zybooks that were listed for the DT regressor but not the DT classifier:

| Parameter   | Default           | Description |  
| ----------- | ----------------- | ----------- |  
| `criterion` | `"squared_error"` | Determines ERROR ESTIMATE used during training. (See above.) |  
| `cpp_alpha` | `0.0`             | Determines $\alpha$ for cost-complexity pruning. (Default does NO pruning.) |  
