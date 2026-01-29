# 3.1: Loss functions for classification

## Loss functions

> A **loss function** quantifies the DIFFERENCE btwn a model's PREDICTIONS and the OBSERVED VALUES (i.e. actual values).

- Lower values indicate better performance (for predictions).
- During model fitting, loss functions are used when choosing weights and params: the goal is to find weights/params that minimize the loss function you're using.
  - e.g. a log reg model is fitted by selecting the weights that minimize the log loss.
- LOW LOSS: GOOD PREDICTIONS.
- HIGH LOSS: BAD PREDICTIONS.

### Loss for classification

- For classification models, two things are accounted for in loss:
  - Did the model predict the right class?
  - Was the model certain (for its correct predictions)?
    - (Anything w/ a prediction probability lower than 1.00 is uncertain.)
- (Even correct classifications can come with a penalty for uncertain predictions. i.e., the model might predict the right class, but with a low prediction probability (certainty).)

## Absolute Loss

> **ABSOLUTE loss** of an instance quantifies the LOSS DUE TO UNCERTAINTY.  
> It considers the DIFFERENCE btwn the PREDICTED PROBABILITY and the OBSERVED (binary) CLASS.

$$L_{abs}(y_i, \hat p_i) = |y_i - \hat p_i|$$

- $y_i$ is the observed class, encoded as `0` or `1`.
- $\hat p_i$ is the predicted probability for the $i^{\text{th}}$ instance.
- The OVERALL abs loss function is given by the AVERAGE ABS LOSS OF ALL INSTANCES.
- "The loss is greater when the predicted probability is further from the observed class."
  - (Because the class is either `0` or `1`, and probs are btwn `0` and `1` as well...ig?)
- Abs loss does NOT consider the predicted class: Whether/not an instance's class was predicted correctly does NOT affect abs loss, bc abs loss only considers the diff btwn PREDICTED PROBABILITY and the observed class. 

## Log loss & likelihood functions

### Likelihood functions

> A **likelihood function** "measures the prob of observed data given a specified statistical distribution and params." (TODO: HUH???)

- For likelihood funcs: The higher the value of the func, the better the model fits&mdash;or would result in&mdash;the observed data.

### Log loss

For log reg models, we oft measure loss w/ a log-likelihood func for "mathematical convenience".

> The **log loss** of an instance is the NEGATIVE LOG-LIKELIHOOD (which comes out as non-negative).

$$L_{log}(y_i, \hat p_i) = -\bigg(y_i\ln(\hat p_i) + (1-y_i) \ln(1-\hat p_i)\bigg)$$

- Why neg log-likelihood? Well-fitting models would have a high log-likelihood and a small negative log-likelihood. So we just use the negative version.
- (Again, the OVERALL log loss is given by the AVERAGE LOG LOSS of all instances.)
- HIGHER PREDICTED PROB => LOWER LOG LOSS.

## Cross-entropy loss

Log loss is great if we're classifying a binary class. (Bc then our output feature can be encoded as `0` or `1`, so the math works out nicely.)

But what if the output feature has two or more classes?!? Enter entropy. 

> **Entropy** is a measure of UNCERTAINTY in a PROB DISTRIBUTION.  
> **Cross-entropy** is a measure of the DIFFERENCE btwn the OBSERVED prob dist and the PREDICTED prob dist of classes.

Consider an output feature $j=1,2,\dots,c$ classes. Let $\mathbf y = [y_{i1}, y_{i2}, \dots, y_{ij}]$, where $y_{ij} = 1$ if instance $i$ belongs to class $j$ (and $0$ if not). Then the cross-entropy loss of an instance $i$ is given by:

$$L_{entropy}(\mathbf y_i, \mathbf {\hat p}_i) = -\mathbf y_i \cdot \ln(\mathbf{\hat p_i}) = \sum_{j=1}^c y_{ij} \ln (\hat p_{ij})$$

- $\mathbf{\hat p_i}$ is instance $i$'s vector of predicted class probabilities.
- Each $y_{ij}$ is an observed distribution, whose predicted distrubtion is $\hat p_{ij}$.

(If the output feature vector is only a length of 2 (meaning there are only two classes), then cross-entropy loss is equivalent to log loss.)

## Hinge loss

> **Hinge loss** uses the DISTANCE an instance is from a DECISION BOUNDARY to measure loss.

Some classif models classify instances relative to a decision boundary. (Whatever tf that means?) Instead of a predicted prob, these models (typically) ret the signed distance that an instance lies from the decision boundary. Hence the outcome feature is either `-1` or `1`, NOT either `0` or `1`.

**Hinge loss** is given by:

$$L_{hinge}(y_i, d_i) = \text{max}(0,1-y_i d_i)$$

- $y_i$ is the observed class for instance $i$, encoded as either `-1` or `1`.
- $d_i$ is the signed distance $i$ is from the decision boundary.

Interpreting hinge loss looks a bit different:

- Correctly classified instances: $0 \le L_{hinge} \le 1$
  - LOWER value for hinge loss => HIGHER distance from the decision boundary.
- Incorrectly classified instances: $L_{hinge} \gt 1$

(Again, the overall hinge loss is the average hing loss of all instances.)

## Loss functions in scikit-learn

- Loss functions come from the `sklearn.metrics` module.

<br>

- Log loss & cross-entropy loss are implemented via `log_loss()`:
  - Params:
    - Array of observed classes (`y_true`)
    - Array of predicted probs (`y_pred`)
  - Implements cross-entropy loss when `y_true` consists of more than two classe.
  - [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)

<br>

- Hinge loss is implemented via `hinge_loss()`
  - Params:
    - Array of observed classes (`y_true`)
      - Elements must consist of one of two integers, where the integer for the positive class is greater than the integer for the negative class.
    - Array of predicted decisions (`pred_discision`)
  - [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html)

