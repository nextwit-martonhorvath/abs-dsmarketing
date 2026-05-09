## 1. Log Loss & CTR Basics

### Click-Through Rate (CTR)

CTR is simply the ratio of clicks to impressions:

$$\text{CTR} = \frac{\text{Number of Clicks}}{\text{Number of Impressions}}$$

An **impression** is every time an ad is shown to a user. A **click** is when the user actually clicks on it. Typical CTR in display/mobile advertising is very low — usually **0.5% to 3%**. This means the vast majority of impressions don't result in clicks, which is important to keep in mind (class imbalance).

### Why Predict CTR?

Avazu connects advertisers (who want clicks) with publishers (who have ad slots). The platform needs to decide **which ad to show** in each slot. This decision is typically made via a real-time auction:

- Advertiser bids a price per click (e.g., €0.50)
- The platform estimates the probability of a click (e.g., 2%)
- The **expected revenue** per impression = bid × predicted CTR = €0.50 × 0.02 = €0.01
- The ad with the highest expected revenue wins the slot

So the task isn't just "will this be clicked?" (binary classification) — it's **"what is the probability of a click?"** The quality of the probability estimate directly affects revenue.

### Log Loss (Binary Cross-Entropy)

Since we care about probability quality, we need a metric that evaluates **how good the predicted probabilities are**, not just whether the binary prediction is right or wrong. That's log loss:

$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

Where:
- $y_i \in \{0, 1\}$ — actual outcome (clicked or not)
- $p_i \in (0, 1)$ — predicted probability of a click
- $N$ — number of observations

**How to read the formula:**
- When $y_i = 1$ (actual click): only the $\log(p_i)$ term survives. If you predicted $p_i = 0.9$, loss is small ($-\log(0.9) = 0.105$). If you predicted $p_i = 0.01$, loss is huge ($-\log(0.01) = 4.6$).
- When $y_i = 0$ (no click): only the $\log(1 - p_i)$ term survives. If you predicted $p_i = 0.1$, loss is small. If you predicted $p_i = 0.99$, loss is huge.

**Key intuition:** Log loss **heavily punishes confident wrong predictions**. Predicting 0.99 for something that doesn't happen is far worse than predicting 0.6 for something that doesn't happen.

### Why Not Accuracy?

If 98% of impressions are not clicked, a model that always predicts "no click" achieves 98% accuracy — but it's completely useless for ranking ads. Log loss forces the model to produce well-calibrated probabilities rather than just getting the majority class right.

### Perfect vs. Baseline Log Loss

- **Perfect model:** log loss = 0 (predicts 1.0 for all clicks, 0.0 for all non-clicks)
- **Naive baseline:** predict the overall CTR for every observation (e.g., always predict 0.17 if 17% of training data are clicks). This gives you a benchmark log loss to beat.

### What to Take Away

- You're predicting a **probability**, not a class label
- Log loss measures how **calibrated** those probabilities are
- The dataset will be **heavily imbalanced** toward non-clicks
- A good baseline to compare against is "predict the average CTR for everything"
