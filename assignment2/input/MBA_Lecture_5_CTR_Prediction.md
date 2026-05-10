---
title: "Online Advertising: CTR Prediction"
course: "Data Science in Online Marketing"
author: "Maarten Soomer"
date: "April 2026"
source: "MBA Lecture 5 - CTR Prediction.pdf"
---

# Online Advertising: CTR Prediction

**Data Science in Online Marketing**  
**Maarten Soomer**  
**April 2026**

---

## Slide 2 - Course Outline

| Date | Topic | Assignment |
|---|---|---|
| 1 april | Recommenders: Content Based | |
| 8 april | Recommenders: Collaborative Filtering | Yes |
| 15 april | Guest Lecture: Recommenders @ Bol | |
| 15 april | Online advertising: Introduction | |
| 22 april | Online advertising: Reinforcement Learning | |
| 29 april | **Online advertising: CTR prediction** | **Yes** |
| 6 may | Online advertising: Media Mix Modeling & Conversion Attribution | |
| 13 may | Revision | |
| 26 may | Digital exam on location | |

---

## Slide 3 - Online ads performance metrics

**Click-through rate (CTR)**

$$
CTR = \frac{\#\text{ click-throughs}}{\#\text{ impressions}}
$$

**Conversion rate (CR)**

$$
CR = \frac{\#\text{ conversions}}{\#\text{ click-throughs}}
$$

Figure on the slide: a marketing funnel showing the flow from target audience to impressions, click-throughs, and conversions. Image credit on slide: *Customer Acquisition Cost, ibiz.wikidot.com*.

---

## Slide 4 - Advertising Networks

The slide illustrates an advertising network connecting advertisers to publishers.

- Advertisers, such as e-commerce websites, connect to an advertising network.
- The advertising network places ads with publishers, such as websites or apps.
- Example ad locations shown on the slide include display advertising placements such as a billboard-style “Your Ad Here” placement and web-page ad units.

---

## Slide 5 - CTR Prediction

**Last week:** Overall CTR estimation for a few versions of an ad.

- How Bayesian updating can improve CTR estimates in the start-up phase.
- How Multi-Armed Bandits can balance exploration versus exploitation.

**Today:** CTR predictions for many ads/impressions with many features, or context.

- Problem framing: algorithm selection and model evaluation.
- Algorithms:
  - Naïve Bayes.
  - Logistic Regression using SGD.

---

## Slide 6 - CTR Prediction - Features

| Visitor | Advertisement | Publisher | Context |
|---|---|---|---|
| Device: phone, tablet, desktop | Advertiser | Website/App | Time of day |
| Operating system | Product category | Ad position | Day of week |
| Location | Ad text | Page contents | Month/season |
| Browsing behaviour | Ad layout/version | ... | Weather |
| ... | ... | | ... |

---

## Slide 7 - CTR Prediction - Features

- CTR prediction involves many categorical features, such as device, operating system, and website.
- Some categorical features have high cardinality: they can take many possible values. For example, `website` can have many distinct values.
- This creates many possible combinations of feature values, each with limited or even no observations.

Example feature sets:

- `Device = {phone, tablet, pc}`
- `Weekday = {Monday, Tuesday, ...}`

Example sparsity question from the slide:

> How often does a Mac user that previously clicked on a Zalando advertisement see version B of our advertisement on nu.nl on a Monday between 20:00 and 21:00?

---

## Slide 8 - CTR Prediction - Feature Interactions

It is important to consider feature interactions.

Example:

- Assume young people respond very well to ads about games.
- But for other age groups, ads about games perform like other types of ads.
- Introduce a cross feature:

$$
x_{young} x_{game}
$$

There are many possible cross features, further increasing data sparsity and dataset size.

Feature interactions can introduce multicollinearity or violate independence conditions. Manual feature selection might be required.

---

## Slide 9 - CTR Prediction Model

Section divider: **CTR Prediction Model**.

---

## Slide 10 - CTR Prediction Model - Framing

- Algorithm selection.
- Evaluation.

---

## Slide 11 - CTR Prediction Model - Algorithm Selection

Important properties of the CTR prediction problem:

- Binary classification model: click/no click.
  - Class imbalance: click rates are typically low.
- Many categorical features with high cardinality.
- Dynamic environment:
  - Large number of observations, or impressions, in a short time period.
  - New observations come in continuously.
  - New ads and websites appear often.

---

## Slide 12 - CTR Prediction Model

**Observation:** Impression.

- Target:

$$
y = \{Click, No\ Click\}
$$

- Features:

$$
X = (x_1, x_2, x_3, \ldots, x_k)
$$

Examples:

- `Device = {mobile, tablet, desktop}`
- `Weekday = {Monday, Tuesday, ...}`
- ...

**What are we interested in?**

Predict whether an impression with given feature values leads to a click or not.

Example classifier workflow shown on the slide:

```python
from sklearn.module import SomeClassifier

model = SomeClassifier()

# Train the model on a training dataset
model.fit(X_train, y_train)

# Predict class labels: Click or No-Click
model.predict(X)
```

---

## Slide 13 - CTR Prediction Model

Typical click rates are very low overall, so the predicted class will almost always be **No Click**.

We are interested in impressions where the click probability is relatively high compared with other impressions.

| Feature | Click probability |
|---|---:|
| Beer ad | 3% |
| Shoe ad | 1% |

**What are we interested in?**

Estimating the probability of a click for a given set of feature values:

$$
P(Y = Click \mid X_1=x_1, X_2=x_2, \ldots, X_k=x_k)
$$

Example probability-prediction workflow shown on the slide:

```python
from sklearn.module import SomeClassifier

model = SomeClassifier()

# Train the model on a training dataset
model.fit(X_train, y_train)

# Predict probability of each class for observations in X
model.predict_proba(X)
```

---

## Slide 14 - CTR Prediction Model - Evaluation

What error metric do we choose for CTR prediction?

- Classification accuracy?

---

## Slide 15 - CTR Prediction Model - Evaluation

What error metric do we choose for CTR prediction?

- Classification accuracy?

Example:

- Suppose we have 10,000 observations with 100 clicks and 9,900 no-clicks.
- Suppose we have a model that always predicts a no-click.

|  | Predicted Click | Predicted No-Click |
|---|---:|---:|
| Actual Click | 0 | 100 |
| Actual No-Click | 0 | 9,900 |

Question on slide: **Accuracy?**

---

## Slide 16 - CTR Prediction Model - Evaluation

Using the model from the previous slide:

|  | Predicted Click | Predicted No-Click |
|---|---:|---:|
| Actual Click | 0 | 100 |
| Actual No-Click | 0 | 9,900 |

$$
Accuracy = 99\%
$$

A model with high accuracy could still classify all observations as **No Clicks**, but this does not tell us anything useful.

---

## Slide 17 - CTR Prediction Model - Evaluation

What error metric do we choose for CTR prediction?

- Mean Squared Error of predicted probabilities.

Example:

- Suppose we have 10,000 observations with 100 clicks and 9,900 no-clicks.
- Suppose we have a model that always predicts a **0% click probability**.

|  | Predicted click probability | Squared error | # observations |
|---|---:|---:|---:|
| Actual Click | 0.0% | 1 | 100 |
| Actual No-Click | 0.0% | 0 | 9,900 |

$$
MSE = \frac{100}{10000}\cdot 1 + \frac{9900}{10000}\cdot 0 = 0.01
$$

A model that always predicts a 0 click probability can still have a low MSE, but it is not useful.

---

## Slide 18 - CTR Prediction Model - Evaluation

What error metric do we choose for CTR prediction?

- Mean Squared Error of predicted probabilities.

We want a model that can correctly identify impressions where clicks are more likely, even if the data is very imbalanced. We especially do not like clicks to have a predicted click probability of 0.

| Feature | Actual Click Rate |
|---|---:|
| Beer ad | 3% |
| Shoe ad | 1% |

We would like a model that correctly predicts that the click rate for the beer ad is higher than the click rate of the shoe ad.

---

## Slide 19 - CTR Prediction Model - Evaluation

What error metric do we choose for CTR prediction?

- **Log Loss**

$$
Log\ Loss = -\left[y \ln(p) + (1-y)\ln(1-p)\right]
$$

where:

- `y = 0` for No Click or `y = 1` for Click.
- `p` is the predicted click probability.

---

## Slide 20 - CTR Prediction Model - Evaluation

For a click (`y = 1`), log loss becomes:

$$
Log\ Loss = -\ln(p)
$$

The slide compares squared error and log loss for a clicked impression as predicted click probability increases. Both errors decrease as `p` approaches 1, but log loss penalizes very low predicted click probabilities much more heavily.

---

## Slide 21 - CTR Prediction Model - Evaluation

What error metric do we choose for CTR prediction?

- **Log Loss**:

$$
Log\ Loss = -\left[y \ln(p) + (1-y)\ln(1-p)\right]
$$

Example:

- Suppose we have 10,000 observations with 100 clicks and 9,900 no-clicks.
- Suppose we have a model that always predicts a **0% click probability**.

|  | Predicted click probability | Log Loss | # observations |
|---|---:|---:|---:|
| Actual Click | 0.0% | ∞ | 100 |
| Actual No-Click | 0.0% | 0 | 9,900 |

$$
Log\ Loss = \frac{100}{10000}\cdot \infty + \frac{9900}{10000}\cdot 0 = \infty
$$

---

## Slide 22 - CTR Prediction Model - Framing

We are interested in estimating the **probability of a click** for a given set of feature values in order to find impressions where the click rate is relatively high.

Log Loss is an error metric on the predicted probabilities that heavily penalizes predictions of a zero probability for observed clicks, and the reverse. This error metric avoids selecting models that always predict a 0 probability because of imbalanced data and low overall click rates.

|  | Actual Click Rate |
|---|---:|
| Beer ad | 3% |
| Shoe ad | 1% |

---

## Slide 23 - CTR Prediction Model - Online Learning

Dynamic environment:

- Large number of observations, or impressions, in a short time period.
- New observations come in continuously.
- New ads and websites appear often.

Therefore: **Online Learning**.

| Batch Learning | Online Learning |
|---|---|
| Train the model on the entire dataset at once. | Update the model observation by observation. |
| Algorithms: Logistic Regression, Decision Tree, Random Forest, ... | Algorithms: Naïve Bayes, Stochastic Gradient Descent, ... |

---

## Slide 24 - CTR Prediction Model - Online Learning

### Batch Learning

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the complete data from the file
df_data = pd.read_csv('data.csv')

model = LogisticRegression()

# Prepare data
# Split train and test set
# Split target variable (y) and features (X)
...

# Train the model on all the training data at once
model.fit(X_train, y_train)
```

### Online Learning

```python
import pandas as pd
from sklearn.linear_model import SGDClassifier

# Prepare reading from file
chunks = pd.read_csv('data.csv', chunksize=1)

model = SGDClassifier()

# Loop over the rows of the file to read and process
for df_row in chunks:
    # Prepare data (row)
    # Split train and test?
    # Split target variable (y) and features (X)
    ...

    # Update the trained model with current row of data
    model.partial_fit(X_row, y_row, ...)
```

---

## Slide 25 - CTR Prediction Model - Framing

- Classification model, click/no-click: predicting click probabilities.
  - Handle class imbalance due to low click rates.
  - Handle many categorical features with high cardinality.
  - Use online training to handle a large number of streaming observations.
- Model evaluation:
  - Log Loss metric.

Algorithms discussed next:

- Naïve Bayes.
- Logistic Regression using Stochastic Gradient Descent.

---

## Slide 26 - Naïve Bayes

Section divider: **Naïve Bayes**.

---

## Slide 27 - Naïve Bayes for CTR Prediction

Outcome:

$$
Y = \{Click, No\ Click\}
$$

Features:

$$
X = (X_1, X_2, X_3, \ldots, X_k)
$$

Examples:

- `Device = {phone, tablet, pc}`
- `Weekday = {Monday, Tuesday, ...}`
- ...

We are interested in estimating:

$$
P(Y = Click \mid X_1=x_1, X_2=x_2, \ldots, X_k=x_k)
$$

Naïve Bayes combines:

- Bayes' Rule.
- Conditional independence assumption.

---

## Slide 28 - Naïve Bayes for CTR Prediction

Starting from Bayes' Rule:

$$
P(Click \mid X_1=x_1, X_2=x_2, \ldots)
= \frac{P(X_1=x_1, X_2=x_2, \ldots \mid Click)P(Click)}{P(X_1=x_1, X_2=x_2, \ldots)}
$$

For the binary outcome click/no-click:

$$
P(Click \mid X_1=x_1, X_2=x_2, \ldots)
=
\frac{P(X_1=x_1, X_2=x_2, \ldots \mid Click)P(Click)}{P(X_1=x_1, X_2=x_2, \ldots \mid Click)P(Click) + P(X_1=x_1, X_2=x_2, \ldots \mid No\ Click)P(No\ Click)}
$$

With the conditional independence assumption:

$$
P(Click \mid X_1=x_1, X_2=x_2, \ldots)
=
\frac{\prod_{i=1}^k P(X_i=x_i \mid Click)P(Click)}{\prod_{i=1}^k P(X_i=x_i \mid Click)P(Click) + \prod_{i=1}^k P(X_i=x_i \mid No\ Click)P(No\ Click)}
$$

---

## Slide 29 - Conditional Independence Assumption

Given the outcome of the trial, the value of feature A is independent of the value of feature B.

Example:

- Features:
  - `Device = {phone, tablet, pc}`
  - `Weekday = {Monday, Tuesday, ...}`
- The assumption implies:

$$
P(Phone \mid Click) = P(Phone \mid Click, Monday) = P(Phone \mid Click, Tuesday)
$$

This does **not** assume that CTR is the same on Monday and Tuesday. It assumes that, among all clicks, the share coming from a phone is the same on Monday and Tuesday.

Given that there is a click, the value of `Device` is independent of the value of `Weekday`.

$$
P(X_1=x_1, X_2=x_2, \ldots, X_k=x_k \mid Y=Click)
= P(X_1=x_1 \mid Y=Click) \times P(X_2=x_2 \mid Y=Click) \times \cdots
= \prod_{i=1}^k P(X_i=x_i \mid Y=Click)
$$

---

## Slide 30 - Naïve Bayes - Example

Outcome:

$$
Y = \{Click, No\ Click\}
$$

Features:

- $X_1 = \{Android, iPhone\}$
- $X_2 = \{Red\ ad\ text, Blue\ ad\ text\}$

Views:

|  | Red | Blue |
|---|---:|---:|
| Android | 4,200 | 2,800 |
| iPhone | 1,800 | 1,200 |

Clicks:

|  | Red | Blue |
|---|---:|---:|
| Android | 42 | 14 |
| iPhone | 12 | 4 |

Totals: **10,000 views** and **72 clicks**.

Observed:

$$
P(Click \mid iPhone, blue) = \frac{4}{1200} = 0.0033
$$

But this is not very reliable because it is based on only 1,200 observations with 4 clicks.

Question on slide: **What does Naïve Bayes do?**

---

## Slide 31 - Naïve Bayes - Example

Naïve Bayes uses all observations it has for iPhones and all observations it has for blue text, independently.

Views:

|  | Red | Blue |
|---|---:|---:|
| Android | 4,200 | 2,800 |
| iPhone | 1,800 | 1,200 |

Clicks:

|  | Red | Blue |
|---|---:|---:|
| Android | 42 | 14 |
| iPhone | 12 | 4 |

Conditional probability table:

| Probability | Click | No Click |
|---|---:|---:|
| $P(Y=y)$ | 72/10000 | 9928/10000 |
| $P(X_1=Android \mid Y=y)$ | 56/72 | 6944/9928 |
| $P(X_1=iPhone \mid Y=y)$ | 16/72 | 2984/9928 |
| $P(X_2=red \mid Y=y)$ | 54/72 | 5946/9928 |
| $P(X_2=blue \mid Y=y)$ | 18/72 | 3982/9928 |

Prediction:

$$
P(Y=Click \mid X_1=iPhone, X_2=blue)
= \frac{P(Y=Click)P(X_1=iPhone \mid Y=Click)P(X_2=blue \mid Y=Click)}{\sum_y P(Y=y)P(X_1=iPhone \mid Y=y)P(X_2=blue \mid Y=y)}
$$

$$
= \frac{\frac{72}{10000}\cdot\frac{16}{72}\cdot\frac{18}{72}}{\frac{72}{10000}\cdot\frac{16}{72}\cdot\frac{18}{72} + \frac{9928}{10000}\cdot\frac{2984}{9928}\cdot\frac{3982}{9928}}
= 0.0033
$$

---

## Slide 32 - Naïve Bayes - Example

The simulation on the slide compares CTR predictors for **iPhone + blue** by SSE as the number of observations `N` increases. The compared predictors are:

- Sample mean.
- Bayesian updating with `Beta(1,99)`.
- Naïve Bayes.

Simulation result stated on the slide:

- Naïve Bayes outperforms the sample mean and Bayesian updating when conditional independence holds.
- This is especially true when the number of observations is small.

Why?

- Sample mean and Bayesian updating only use approximately 120 of every 1,000 observations.
- Naïve Bayes uses all observations for iPhone and all observations for blue.

---

## Slide 33 - Naïve Bayes - Review

Pros:

- Fast to train through online learning.
- Very scalable.
- Not sensitive to irrelevant features.
- Widely used and gives good results in many different applications.

Cons:

- Assumes conditional independence of the features.
- Manual definition of feature interactions and feature selection required.

Note: Naïve Bayes is not necessarily a Bayesian method, but it can be if the parameters are estimated by Bayesian updating.

---

## Slide 34 - Logistic Regression

Section divider: **Logistic Regression**.

---

## Slide 35 - Logistic Regression

Logistic regression prediction function:

$$
Y = \frac{1}{1 + e^{-(\beta_1x_1 + \beta_2x_2 + \cdots)}}
$$

where:

- Outcome: $Y = 0$ or $Y = 1$, corresponding to No Click or Click.
- Features: $X = (x_1, x_2, x_3, \ldots, x_k)$.
- Coefficients: $\beta_1, \beta_2, \beta_3, \ldots, \beta_k$.

The logistic regression function transforms feature values into a predicted probability of the outcome, with a value between 0 and 1.

We want to find coefficient values $\beta_0, \beta_1, \beta_2, \ldots$ that fit the data best.

Logistic regression is the classification equivalent of linear regression.

---

## Slide 36 - Logistic Regression - Categorical Features

In logistic regression, feature values need to be numeric.

- A categorical feature can be represented as a series of binary variables, also called dummy variables, for each possible value.
- This is done using **one-hot encoding**.

Example feature values:

- `Device = {phone, tablet, pc}`
- `Weekday = {Monday, Tuesday, ...}`

Variable mapping:

| Variable | Feature | Value |
|---|---|---|
| $X_1$ | Device | phone |
| $X_2$ | Device | tablet |
| $X_3$ | Device | pc |
| $X_4$ | Weekday | Monday |
| $X_5$ | Weekday | Tuesday |
| $X_6$ | Weekday | Wednesday |
| ... | ... | ... |

An observation for a PC on Tuesday is represented as:

$$
x_1 = 0,\ x_2 = 0,\ x_3 = 1,\ x_4 = 0,\ x_5 = 1,\ x_6 = 0,\ldots
$$

---

## Slide 37 - Logistic Regression - Categorical Features

Example one-hot encoding workflow:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

# Fit the encoder to the data in X_train and transform the data
X_train_encoded = encoder.fit_transform(X_train)

# Use the encoder to transform the data in X_test
X_test_encoded = encoder.transform(X_test)
```

---

## Slide 38 - Logistic Regression - Example

Outcome:

$$
Y = \{Click, No\ Click\}
$$

Features:

- $X_1 = \{Android, iPhone\}$
- $X_2 = \{Red\ ad\ text, Blue\ ad\ text\}$

Views:

|  | Red | Blue |
|---|---:|---:|
| Android | 4,200 | 2,800 |
| iPhone | 1,800 | 1,200 |

Clicks:

|  | Red | Blue |
|---|---:|---:|
| Android | 42 | 14 |
| iPhone | 12 | 4 |

Totals: **10,000 views** and **72 clicks**.

Example one-hot encoded observations:

| n | Click | Android | iPhone | red | blue |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 0 | 1 | 0 |
| 2 | 0 | 1 | 0 | 0 | 1 |
| 3 | 1 | 0 | 1 | 1 | 0 |
| 4 | 0 | 0 | 1 | 0 | 1 |

Here, `Click` is the target variable `Y`; the dummy variables are the feature matrix `X`.

---

## Slide 39 - Logistic Regression - Example

Estimated coefficients:

| Variable | Coefficient |
|---|---:|
| Android | 0.4082 |
| iPhone | 0 |
| red | -5.0035 |
| blue | -5.7012 |

Prediction for iPhone with blue ad text:

$$
P(Click \mid iPhone, blue)
= \frac{1}{1 + e^{-(\beta_1x_1 + \beta_2x_2 + \cdots)}}
= \frac{1}{1 + e^{-(0 - 5.7012)}}
= 0.0033
$$

---

## Slide 40 - But...

For CTR prediction, we typically have:

- Many categorical features, resulting in an enormous number of dummy variables, which can be multicollinear.
- Millions of observations.

Common estimation techniques for logistic regression cannot handle such a large matrix with multicollinearity.

---

## Slide 41 - Stochastic Gradient Descent

Stochastic Gradient Descent is:

- An online learning algorithm.
- A method that can be used to fit a logistic regression model with many categorical features and many observations efficiently.

---

## Slide 42 - Stochastic Gradient Descent

Minimize the log loss error:

$$
\min \frac{1}{N}\sum_t -\left[y_t\ln(p_t) + (1-y_t)\ln(1-p_t)\right]
$$

where the prediction $p_t$ is given by the logistic regression function:

$$
p_t := \frac{1}{1 + e^{-(\beta_1x_1 + \beta_2x_2 + \cdots)}}
$$

We want to find the coefficients $\beta_1, \beta_2, \ldots$ that minimize:

$$
\min_{\beta_1,\beta_2,\ldots}\frac{1}{N}\sum_t -\left[y_t\ln\left(\frac{1}{1+e^{-(\beta_1x_1+\beta_2x_2+\cdots)}}\right) + (1-y_t)\ln\left(1-\frac{1}{1+e^{-(\beta_1x_1+\beta_2x_2+\cdots)}}\right)\right]
$$

---

## Slide 43 - Stochastic Gradient Descent

Update coefficient values for observation `t`:

$$
\beta_i = \beta_i - \eta_t(p_t-y_t)x_{t,i}
$$

where:

- $\beta_i$ is the value of coefficient `i`.
- $\eta_t$ is a non-increasing learning rate between 0 and 1, for example:

$$
\eta_t = \frac{1}{\sqrt{t}}
$$

The update is determined by the size of the error $p_t - y_t$ in observation `t`:

- When we observe a click, the coefficient value is increased.
- When we observe a no-click, the coefficient value is decreased.
- There is no change in value for coefficients where the variable value is zero: $x_i = 0$ in observation `t`.

---

## Slide 44 - Stochastic Gradient Descent - Example

Outcome:

$$
Y = \{Click, No\ Click\}
$$

Features:

- $X_1 = \{Android, iPhone\}$
- $X_2 = \{Red\ ad\ text, Blue\ ad\ text\}$

Assume all coefficients are initialized to zero:

$$
\beta_{android} = \beta_{iPhone} = \beta_{red} = \beta_{blue} = 0
$$

Initial CTR prediction before any observation:

$$
p_1 = \frac{1}{1+e^{-(\beta_1x_1+\beta_2x_2+\cdots)}} = \frac{1}{1+e^0} = \frac{1}{2}
$$

This gives a prediction of 0.50 for all feature values.

Learning rate:

$$
\eta_t = \frac{1}{t},\quad \eta_1 = 1
$$

Initial predictions:

|  | Red | Blue |
|---|---:|---:|
| Android | 0.50 | 0.50 |
| iPhone | 0.50 | 0.50 |

---

## Slide 45 - Stochastic Gradient Descent - Example

First observation: an impression on an Android device with red text, with **No Click**.

$$
y_1 = 0,\quad p_1 = \frac{1}{2}
$$

Coefficient updates based on the first observation:

$$
\beta_{android} = 0 - \left(\frac{1}{2} - 0\right) \times 1 = -\frac{1}{2}
$$

$$
\beta_{iPhone} = 0 - \left(\frac{1}{2} - 0\right) \times 0 = 0
$$

$$
\beta_{red} = 0 - \left(\frac{1}{2} - 0\right) \times 1 = -\frac{1}{2}
$$

$$
\beta_{blue} = 0 - \left(\frac{1}{2} - 0\right) \times 0 = 0
$$

Predictions before update:

|  | Red | Blue |
|---|---:|---:|
| Android | 0.50 | 0.50 |
| iPhone | 0.50 | 0.50 |

Predictions after update:

|  | Red | Blue |
|---|---:|---:|
| Android | 0.27 | 0.38 |
| iPhone | 0.38 | 0.50 |

---

## Slide 46 - Stochastic Gradient Descent - Example

Training on more observations. The prediction for the current observation uses the coefficient values from just before the update.

$$
p = \frac{1}{1+e^{-(\beta_1x_1+\beta_2x_2+\cdots)}}
$$

| t | η = 1/√t | β1 | β2 | β3 | β4 | x1 | x2 | x3 | x4 | p | Click |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1 | 0 | 1 | 0 | 0.50 | 0 |
| 2 | 0.71 | -0.50 | 0.00 | -0.50 | 0.00 | 1 | 0 | 1 | 0 | 0.27 | 0 |
| 3 | 0.58 | -0.69 | 0.00 | -0.69 | 0.00 | 0 | 1 | 0 | 1 | 0.50 | 0 |
| 4 | 0.50 | -0.69 | -0.29 | -0.69 | -0.29 | 1 | 0 | 1 | 0 | 0.20 | 1 |
| 5 | 0.45 | -0.29 | -0.29 | -0.29 | -0.29 | 1 | 0 | 1 | 0 | 0.36 | 0 |
| 6 | 0.41 | -0.45 | -0.29 | -0.45 | -0.29 | 0 | 1 | 0 | 1 | 0.36 | 0 |
| 7 | 0.38 | -0.45 | -0.44 | -0.45 | -0.44 | 0 | 1 | 0 | 1 | 0.30 | 0 |
| 8 | 0.35 | -0.45 | -0.55 | -0.45 | -0.55 | 1 | 0 | 0 | 1 | 0.27 | 0 |
| 9 | 0.33 | -0.55 | -0.55 | -0.45 | -0.64 | 0 | 1 | 1 | 0 | 0.27 | 0 |
| 10 | 0.32 | -0.55 | -0.64 | -0.54 | -0.64 | 0 | 1 | 0 | 1 | 0.22 | 0 |
| 11 | 0.30 | -0.55 | -0.71 | -0.54 | -0.71 | 1 | 0 | 1 | 0 | 0.25 | 0 |
| 12 | 0.29 | -0.62 | -0.71 | -0.62 | -0.71 | 0 | 1 | 1 | 0 | 0.21 | 0 |
| 13 | 0.28 | -0.62 | -0.77 | -0.68 | -0.71 | 1 | 0 | 0 | 1 | 0.21 | 1 |
| 14 | 0.27 | -0.40 | -0.77 | -0.68 | -0.49 | 1 | 0 | 0 | 1 | 0.29 | 0 |
| 15 | 0.26 | -0.48 | -0.77 | -0.68 | -0.57 | 1 | 0 | 0 | 1 | 0.26 | 0 |
| 16 | 0.25 | -0.55 | -0.77 | -0.68 | -0.64 | 1 | 0 | 1 | 0 | 0.23 | 0 |
| 17 | 0.24 | -0.60 | -0.77 | -0.73 | -0.64 | 0 | 1 | 1 | 0 | 0.18 | 0 |
| 18 | 0.24 | -0.60 | -0.81 | -0.78 | -0.64 | 1 | 0 | 0 | 1 | 0.22 | 0 |
| 19 | 0.23 | -0.66 | -0.81 | -0.78 | -0.69 | 0 | 1 | 1 | 0 | 0.17 | 0 |
| 20 | 0.22 | -0.66 | -0.85 | -0.82 | -0.69 | 0 | 1 | 1 | 0 | 0.16 | 0 |

---

## Slide 47 - Example

The slide shows two line charts based on the SGD example:

- **Coefficient values**: coefficient paths for β1, β2, β3, and β4 over observations 1 to 20. The coefficients generally become more negative over time, with jumps after clicked observations.
- **Predicted CTR**: predicted CTR across observations 1 to 20. The predicted CTR starts around 0.50 and declines over time, with increases around observations where clicks are observed.

---

## Slide 48 - Stochastic Gradient Descent - Efficiency

- Every coefficient is updated independently of the others, based only on the most recent observation.
- This is not optimal, but it is very efficient with many categorical features.

Example:

- Assume 10 categorical features with 100 possible values each.
- One-hot encoding gives 1,000 dummy variables.
- But each observation has only 10 non-zero variables.
- With SGD, we only need to update the values for 10 coefficients per observation.

---

## Slide 49 - Stochastic Gradient Descent - Extensions

Feature-specific learning rate:

- It is common to use a specific learning rate for each coefficient separately.
- This learning rate is only updated when the variable is observed, meaning $x_i = 1$ at time `t`.
- This keeps learning fast for feature values that have few observations, even if other features have many observations.

Instead of a single learning rate $\eta_t$ for observation `t`, we use learning rates $\eta_{t,i}$ for each variable `i`.

Example:

$$
\eta_{t,i} = \frac{1}{\text{count of observations with } x_i = 1}
$$

---

## Slide 50 - Stochastic Gradient Descent - Mini-Batches

In practice, coefficients are typically not updated after every new observation, but after every `B` new observations.

Advantages:

- Smoother convergence.
- More computationally efficient, since vector multiplications tend to be faster than for loops.

---

## Slide 51 - Logistic Regression using SGD - Review

Pros:

- Fast to train through online learning.
- Very scalable.

Cons:

- Manual definition of feature interactions and feature selection required.

---

## Slide 52 - Wrap Up

Section divider: **Wrap Up**.

---

## Slide 53 - Reinforcement Learning: Contextual Bandits

Contextual bandits combine CTR prediction with many features, or context, with Multi-Armed Bandits.

- Online learning of CTR of an ad given the context, or features.
- A policy considering the confidence of the CTR prediction to balance exploration and exploitation.

---

## Slide 54 - Feature Interactions: Factorization Machines

Apply factorization to model feature interactions.

Regression:

$$
\beta_1x_1 + \beta_2x_2 + \cdots + \beta_{1,2}x_1x_2 + \beta_{1,3}x_1x_3 + \cdots
$$

Factorization machines:

$$
\beta_1x_1 + \beta_2x_2 + \cdots + v_1v_2x_1x_2 + v_1v_3x_1x_3 + \cdots
$$

where $v_i$ is a vector of `L` latent factor scores, with $L \ll N$ features.

Instead of estimating $N^2$ coefficients, we need to estimate only $N \times L$ latent factor scores.

$$
\beta_{1,2} = v_1v_2
= \begin{bmatrix} v_{1,1} & v_{1,2} & \cdots & v_{1,L} \end{bmatrix}
\begin{bmatrix} v_{2,1} \\ v_{2,2} \\ \vdots \\ v_{2,L} \end{bmatrix}
$$

---

## Slide 55 - In Practice

| Study | Application | Model/Method | Additional notes |
|---|---|---|---|
| McMahan et al. (2013) | Predicting CTR for Google AdWords | Logistic regression using SGD variant | Feature-specific learning rates used for confidence estimates, connected to reinforcement learning. |
| Graepel et al. (2010) | Predicting CTR for Bing Sponsored Search | Probit regression using Bayesian online learning |  |
| He et al. (2014) | Predicting CTR for Facebook ads | Logistic regression using SGD | Encoding of features using boosted tree models. |

---

## Slide 56 - Today's Course Material

- Lecture notes.
- Homework Exercises.
- Computer Assignment 2.

---

## Slide 57 - References

- Graepel, Candela, Borchert & Herbrich (2010), *Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising on Microsoft's Bing Search Engine*, Proceedings of the 27th International Conference on Machine Learning.
- He, et al. (2014), *Practical Lessons from Predicting Clicks on Ads at Facebook*, ADKDD.
- McMahan, et al. (2013), *Ad Click Prediction: A View From The Trenches*, KDD.

References are included as background information only. They do not need to be studied for the exam.
