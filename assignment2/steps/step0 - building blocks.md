## Assignment Walkthrough: CTR Prediction for Avazu

### What You're Building

A system that predicts the **probability** an ad impression becomes a click, using two specific models trained via online learning on ~40M records.

---

### Theoretical Building Blocks to Study

**1. Click-Through Rate (CTR) Prediction**
- What CTR means in advertising (clicks / impressions)
- Why predicting *probability* matters (not just click/no-click classification) — this drives ad auction bidding and revenue optimization

**2. Logistic Regression with Stochastic Gradient Descent (SGD)**
- Logistic regression as a linear model for probability estimation (sigmoid function)
- How SGD differs from batch gradient descent — updating weights after each observation or mini-batch
- Learning rate selection and decay schedules
- Regularization (L1/L2) and why it matters with high-dimensional sparse features
- Scikit-learn's `SGDClassifier` with `loss='log_loss'`

**3. Naïve Bayes**
- Bayes' theorem and the "naïve" conditional independence assumption
- Why **Multinomial** or **Bernoulli** Naïve Bayes fits categorical features (not Gaussian)
- `partial_fit()` for incremental/online learning
- Laplace smoothing

**4. Online Learning / Incremental Learning**
- Why you can't load 40M rows into memory at once
- The `partial_fit()` API — feeding data observation-by-observation or in mini-batches
- Reading CSV in chunks (`pd.read_csv(..., chunksize=...)`)
- Differences from batch learning: no re-reading of past data, single pass or few passes

**5. Feature Engineering for Categorical Variables**
- All features are **nominal categorical** (even numeric-looking ones)
- One-hot encoding / hashing trick for high-cardinality categoricals
- `HashingVectorizer` or `FeatureHasher` — critical for memory-efficient encoding of features you haven't seen yet
- Feature selection: which columns are informative vs. noise vs. too high-cardinality to be useful

**6. Log Loss (Binary Cross-Entropy)**
- The evaluation metric: $\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i)\log(1-p_i)]$
- Why log loss penalizes confident wrong predictions heavily
- How to compute it incrementally (running sum) without storing all predictions

**7. Evaluation on Unseen Data**
- Data is **chronologically ordered** — this hints at a temporal train/test split (not random)
- Why random splits would cause data leakage in time-series-like ad data
- Prequential evaluation (test-then-train) as an alternative

**8. Class Imbalance**
- CTR datasets are heavily skewed (typically 1-5% click rate)
- How this affects model training and evaluation
- Why accuracy is misleading and log loss is appropriate

---

### Practical Concepts to Understand

- **Chunked file reading** — streaming large CSVs without loading into memory
- **Computation time comparison** — SGD vs. Naïve Bayes have different computational profiles per update
- **Reproducibility** — setting random seeds, documenting chunk sizes and parameter choices

### Suggested Study Order

1. Log loss & CTR basics → 2. Online learning concept → 3. Logistic Regression + SGD → 4. Naïve Bayes → 5. Feature hashing for categoricals → 6. Chunked data processing → 7. Temporal evaluation strategy