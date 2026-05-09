## 7. Temporal Evaluation Strategy

### Why Not Random Splits?

In a typical ML workflow, you'd shuffle the data and split randomly (e.g., 80/20 train/test). **Don't do that here.** Two reasons:

**1. Data leakage.** The Avazu data is chronologically ordered — 10 days of ad impressions in time sequence. If you randomly split, your training set will contain observations from Day 8 and your test set will contain observations from Day 3. The model trains on "future" data to predict the "past." In production, you never have future data. This inflates performance unrealistically.

**2. Temporal patterns change.** User behavior, popular apps, and trending sites shift over time. A model deployed on Monday was trained on last week's data. Your evaluation should mimic this: train on earlier data, evaluate on later data.

### The Three Evaluation Options

#### Option A: Fixed Temporal Split

Split the file at a specific point in time. Since the data covers 10 days:

- **Train:** Days 1–8 (~80% of data)
- **Test:** Days 9–10 (~20% of data)

You can determine the split by the `hour` column. Or, since the file is in chronological order, simply count rows:

- ~40M rows total → train on first ~32M, test on last ~8M
- With `chunksize=50,000` → train on chunks 0–639, test on chunks 640+

**Pros:** Clean separation. Easy to explain. Mirrors production setup.
**Cons:** You "waste" 20% of data for training. Test set covers only a narrow time window.

#### Option B: Prequential Evaluation (Test-Then-Train)

For each chunk: **first** predict and evaluate, **then** train on it:

```
Chunk 1: predict (random, no training yet) → evaluate → train
Chunk 2: predict (trained on chunk 1) → evaluate → train
Chunk 3: predict (trained on chunks 1-2) → evaluate → train
...
```

Every observation is predicted *before* the model has seen it — so it's always "unseen." The model continuously improves as it processes more data.

**Pros:** Uses all data for both training and evaluation. More statistically efficient. Standard approach in online learning literature.
**Cons:** Early predictions are terrible (untrained model), inflating overall log loss. Mixes train and evaluation somewhat. You can mitigate by discarding the first N chunks from the evaluation metric ("burn-in" period).

#### Option C: Day-by-Day Evaluation

A hybrid: for each day, evaluate on that day's data using a model trained on all previous days:

```
Train on Day 1 → Evaluate on Day 2
Train on Days 1-2 → Evaluate on Day 3
Train on Days 1-3 → Evaluate on Day 4
...
```

This gives you a **time series of log loss values** — you can see how performance evolves and whether the model degrades or improves over time.

**Pros:** Rich insight into temporal stability. Very realistic.
**Cons:** More complex to implement. Requires identifying day boundaries in the chunked stream.

### Which to Choose?

Any of these is defensible. The key is that you **justify your choice** in the report.

- **Option A** is the simplest and most conventional. If you're unsure, go with this.
- **Option B** is the most natural for online learning and maximizes data usage.
- **Option C** gives the most insight but is the most work.

You can also combine: use Option A as your primary metric and show a prequential learning curve as supplementary analysis.

### What Makes Temporal Evaluation "Efficient"?

The assignment says to evaluate "in an efficient way." This means:

1. **Don't store all predictions.** Use the incremental log loss computation (running sum + count) from step 6.
2. **Don't re-read the file.** Ideally, do training and evaluation in a single pass through the data.
3. **Don't load the test set into memory.** Evaluate chunk-by-chunk, same as training.

With the fixed split, a single-pass approach looks like:

```
for i, chunk in enumerate(reader):
    X, y = encode(chunk)
    if i < train_cutoff:
        model.partial_fit(X, y, classes=[0,1])
    else:
        p = model.predict_proba(X)[:, 1]
        accumulate_log_loss(y, p)
```

One pass. One reader. Constant memory. That's efficient.

### Comparing Two Models Fairly

You're evaluating both SGD Logistic Regression and Naïve Bayes. For a fair comparison:

1. **Same split.** Both models must be evaluated on exactly the same test data.
2. **Same features.** Use the same feature set and encoding for both (though `FeatureHasher` settings may differ — `non_negative=True` for NB).
3. **Same chunk size.** Especially for timing comparisons.
4. **Fresh reader per model.** The CSV iterator is consumed after one pass. Create a new `pd.read_csv(..., chunksize=N)` for each model.

### Baseline to Compare Against

Always compute a **naive baseline** log loss — predicting the overall average CTR for every impression. This tells you whether your models are actually learning anything:

1. During training chunks, track the running click rate: $\text{CTR} = \frac{\text{total clicks}}{\text{total impressions}}$
2. Use that as $p$ for every test observation
3. Compute log loss

If your model's log loss isn't meaningfully lower than this baseline, the features or model aren't adding value.

### Summary: What to Report

For the evaluation section of your report:

| Question | What to discuss |
|----------|----------------|
| How did you split? | Temporal split at day X / prequential / etc. |
| Why that approach? | Avoids data leakage, mimics production |
| Why not random? | Would leak future information |
| How did you compute log loss? | Incrementally, running sum, with clipping |
| What's the baseline? | Naive CTR prediction → log loss of Y |
| SGD log loss vs. NB log loss? | Model A achieved X, Model B achieved Y |
| SGD time vs. NB time? | Model A took X seconds, Model B took Y seconds |