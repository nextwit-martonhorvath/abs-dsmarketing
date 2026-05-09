

## 6. Chunked Data Processing

### Why Chunks?

The Avazu `train.gz` file is ~1.1 GB compressed, ~6.5 GB uncompressed, with ~40 million rows. Loading it all with `pd.read_csv('train')` would try to create a DataFrame using 10+ GB of RAM (and much more after feature encoding). Most machines will either slow to a crawl or crash.

Chunked reading solves this: you load a small piece, process it, discard it, and load the next.

### `pd.read_csv` with `chunksize`

Passing `chunksize=N` makes `read_csv` return an **iterator** instead of a DataFrame. Each iteration yields a DataFrame of N rows:

```python
reader = pd.read_csv('train.gz', chunksize=50000)

for chunk in reader:
    # chunk is a regular DataFrame with 50,000 rows
    # (last chunk may be smaller)
    process(chunk)
    # chunk is garbage-collected after this iteration
```

Only one chunk is in memory at a time. With `chunksize=50000`, you'll iterate ~800 times to cover 40M rows.

### Reading Compressed Files

`pd.read_csv` can read `.gz` files directly — no need to decompress first:

```python
reader = pd.read_csv('train.gz', chunksize=50000)
```

This is slower than reading uncompressed (CPU cost of decompression) but saves disk space. If you have the disk space, decompressing once upfront and reading the uncompressed file is faster for repeated passes.

### The Full Processing Loop

Here's the conceptual structure of how everything fits together:

```
for each chunk:
    1. Read chunk (DataFrame of N rows)
    2. Separate target: y = chunk['click']
    3. Select feature columns
    4. (Optional) Engineer features (e.g., extract hour-of-day)
    5. Convert to feature dicts
    6. Hash features: X = hasher.transform(dicts)
    7. Update model: model.partial_fit(X, y, classes=[0,1])
```

Steps 1–7 repeat until the file is exhausted. The model accumulates learning across all chunks.

### Train/Test Split with Chunks

Since the data is **chronologically ordered** and you need to evaluate on unseen data, you have two main approaches:

**Approach A — Fixed temporal split:**
Use the first ~80% of chunks for training, the last ~20% for evaluation. Since the file has ~40M rows:
- Rows 1–32M → train (chunks 1–640 if chunksize=50,000)
- Rows 32M–40M → test (chunks 641–800)

You'd track chunk number and switch from `partial_fit` to evaluation:

```
for i, chunk in enumerate(reader):
    X, y = encode(chunk)
    if i < split_point:
        model.partial_fit(X, y, classes=[0,1])
    else:
        predictions = model.predict_proba(X)[:, 1]
        accumulate_log_loss(y, predictions)
```

**Approach B — Prequential (test-then-train):**
For each chunk: first predict (evaluate), then train. This uses all data for both training and evaluation, with each observation always being "unseen" at prediction time:

```
for chunk in reader:
    X, y = encode(chunk)
    predictions = model.predict_proba(X)[:, 1]  # test first
    accumulate_log_loss(y, predictions)
    model.partial_fit(X, y, classes=[0,1])       # then train
```

The first chunk's evaluation will be random (untrained model), but this washes out over 40M rows. This approach is more data-efficient but mixes training and evaluation, which some consider less clean.

### Computing Log Loss Incrementally

You don't want to store 40M predictions in memory. Instead, keep a running sum:

$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

Maintain two accumulators:

```
total_loss = 0.0  # running sum of per-sample losses
total_count = 0   # running count of observations

for chunk in test_chunks:
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-15, 1 - 1e-15)  # avoid log(0)
    chunk_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    total_loss += chunk_loss
    total_count += len(y)

log_loss = total_loss / total_count
```

The `np.clip` is important — if the model predicts exactly 0.0 or 1.0, `log(0)` is negative infinity, which ruins everything.

### Chunk Size Considerations

| Chunk Size | Iterations (40M rows) | Memory per chunk | Notes |
|-----------|----------------------|-----------------|-------|
| 1,000 | 40,000 | ~tiny | Slow (Python loop overhead dominates) |
| 10,000 | 4,000 | ~few MB | Reasonable |
| 50,000 | 800 | ~20-30 MB | Good balance |
| 100,000 | 400 | ~50-60 MB | Fast, still manageable |
| 500,000 | 80 | ~200+ MB | Fast but heavier memory use |

The chunk size affects:
- **Speed**: larger = fewer iterations = less Python overhead = faster total time
- **Memory**: larger = more RAM per chunk (both raw data and encoded sparse matrix)
- **SGD learning dynamics**: larger mini-batches = smoother gradients but fewer weight updates per pass. For 40M rows with chunksize=50,000, that's still 800 updates — plenty.

For Naïve Bayes, chunk size doesn't affect the final model at all (counting is order-independent). For SGD, very large chunks mean fewer gradient updates, which could slightly affect convergence.

### Timing Your Models

The assignment asks you to compare computation times. Wrap the training loop:

```python
import time

start = time.time()
for chunk in reader:
    # ... encode and partial_fit ...
elapsed = time.time() - start
```

Do this separately for SGD and Naïve Bayes using the same chunk size, so the comparison is fair. Expect Naïve Bayes to be faster per chunk (counting vs. gradient computation).

### Common Pitfalls

1. **Forgetting `classes=[0,1]` on the first `partial_fit` call.** The model won't know both classes exist. Pass it every time — it's ignored after the first call.

2. **Not resetting the reader.** The `pd.read_csv(..., chunksize=N)` iterator is consumed after one pass. If you need to read the file again (e.g., for a second model), create a new reader.

3. **Feature engineering inside the loop.** Keep it lightweight — complex transformations on every chunk will dominate your runtime.

4. **Not clipping probabilities before log loss.** A single `log(0)` produces `inf` and corrupts your entire metric.

5. **Using the wrong hasher setting for MultinomialNB.** Remember `non_negative=True` — otherwise you get negative feature values and an error.