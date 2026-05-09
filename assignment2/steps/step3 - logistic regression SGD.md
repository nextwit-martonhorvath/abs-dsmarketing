## 3. Logistic Regression with Stochastic Gradient Descent (SGD)

### Logistic Regression — The Core Idea

Logistic regression models the **probability** of a binary outcome. It takes a linear combination of features and passes it through the **sigmoid function** to produce a value between 0 and 1:

$$p(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

Where:
- $x$ = feature vector (one impression's encoded features)
- $w$ = weight vector (one weight per feature — the model learns these)
- $b$ = bias/intercept term
- $\sigma(\cdot)$ = sigmoid function

The sigmoid squashes any real number into $(0, 1)$, which gives us a valid probability.

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| $z$ (linear output) | $\sigma(z)$ (probability) |
|---|---|
| $-\infty$ | → 0 |
| 0 | 0.5 |
| $+\infty$ | → 1 |

When $w^Tx + b$ is large and positive → high predicted probability of click. Large and negative → low probability. The model learns which features push the prediction in which direction.

### How the Model Learns — The Loss Function

The model is trained by minimizing **log loss** (the same metric you'll evaluate with). For a single observation:

$$L(w) = -\left[y \log(\hat{p}) + (1-y) \log(1-\hat{p})\right]$$

This is differentiable, which means we can use **gradient descent** to find the weights $w$ that minimize it.

### Batch Gradient Descent vs. SGD

**Batch gradient descent** computes the gradient using *all* training data:

$$w_{t+1} = w_t - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla L(w_t, x_i, y_i)$$

This requires all N observations in memory and is expensive per step. Not feasible for 40M rows.

**Stochastic Gradient Descent (SGD)** approximates this by using *one observation* (or a mini-batch) at a time:

$$w_{t+1} = w_t - \eta \cdot \nabla L(w_t, x_t, y_t)$$

The gradient for logistic regression on a single sample is:

$$\nabla L = (\hat{p}_t - y_t) \cdot x_t$$

That's it — the prediction error times the feature vector. Simple, fast, and requires only one row in memory.

**Trade-off:** Each SGD update is a noisy estimate of the true gradient direction. The path to the optimum is zigzaggy rather than smooth. But with 40M samples, you have so many updates that it converges well in practice.

### Learning Rate ($\eta$)

The learning rate controls step size. This is the most important hyperparameter for SGD:

- **Too large:** the model overshoots and diverges (loss increases)
- **Too small:** the model learns too slowly, may not converge in one pass
- **Just right:** smooth convergence

**Learning rate schedules** reduce $\eta$ over time, which is standard practice:

- Start with a larger $\eta$ (e.g., 0.01) for fast initial learning
- Decrease it as training progresses for fine-grained convergence

Scikit-learn's `SGDClassifier` supports several schedules via the `learning_rate` parameter:
- `'optimal'` (default): $\eta_t = \frac{1}{\alpha \cdot (t_0 + t)}$ — automatically tuned, good default
- `'constant'`: fixed rate, you set `eta0`
- `'invscaling'`: $\eta_t = \frac{\eta_0}{t^{p}}$ — controlled by `eta0` and `power_t`
- `'adaptive'`: keeps $\eta$ constant until loss stalls, then halves it

### Regularization

With feature hashing you'll have thousands or tens of thousands of features. Regularization prevents overfitting by penalizing large weights:

- **L2 (Ridge):** adds $\frac{\alpha}{2} \|w\|_2^2$ to the loss. Shrinks all weights toward zero. Default in `SGDClassifier`.
- **L1 (Lasso):** adds $\alpha \|w\|_1$ to the loss. Drives some weights exactly to zero (sparsity / automatic feature selection).
- **Elastic Net:** combination of L1 and L2.

The strength is controlled by the `alpha` parameter. Higher $\alpha$ = stronger regularization = simpler model.

### In Scikit-learn: `SGDClassifier`

The key configuration for this assignment:

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(
    loss='log_loss',        # logistic regression (not SVM)
    penalty='l2',           # regularization type
    alpha=0.0001,           # regularization strength
    learning_rate='optimal', # learning rate schedule
    random_state=42
)
```

- `loss='log_loss'` is what makes this logistic regression (without it, `SGDClassifier` defaults to a linear SVM)
- Use `partial_fit(X_chunk, y_chunk, classes=[0, 1])` for online learning
- Use `predict_proba(X)` to get probability estimates (needed for log loss evaluation)

### Parameters You'll Need to Justify

For the report, you need to explain why you chose:

1. **`alpha`** — regularization strength. Try a few values (e.g., 1e-3, 1e-4, 1e-5) and compare log loss
2. **`learning_rate`** — schedule type. `'optimal'` is a safe default but you could experiment
3. **`penalty`** — L1 vs. L2. L1 is interesting if you want implicit feature selection
4. **Mini-batch size** — affects convergence speed and noise level

### Why SGD + Logistic Regression Fits This Problem

- Outputs calibrated probabilities (what the assignment asks for)
- Scales to 40M rows via online learning
- Works with sparse, high-dimensional feature representations (feature hashing)
- Each update is $O(d)$ where $d$ = number of features — very fast