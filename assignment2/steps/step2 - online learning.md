## 2. Online Learning / Incremental Learning

### The Problem

The Avazu dataset is ~6.5 GB uncompressed with ~40 million rows. Loading it all into memory at once would require tens of gigabytes (especially after feature encoding). Most laptops can't handle that. Even if they could, traditional "batch" learning (fit on all data at once) would be extremely slow.

### Batch vs. Online Learning

**Batch learning** (the "normal" way):
1. Load all data into memory
2. Run the learning algorithm over the entire dataset
3. Repeat for multiple passes (epochs) until convergence
4. Model sees all data simultaneously

**Online learning** (what this assignment requires):
1. Read a small chunk of data (one row or a mini-batch)
2. Update the model with just that chunk
3. Discard the chunk from memory
4. Read the next chunk, repeat
5. Model sees data **sequentially**, one piece at a time

### Why Online Learning Works

Both Logistic Regression (with SGD) and Naïve Bayes have a mathematical property that makes this possible: their update rules are **decomposable**.

- **SGD**: each weight update only needs the current observation's gradient, not the full dataset's gradient. The update rule is:

$$w_{t+1} = w_t - \eta \cdot \nabla L(w_t, x_t, y_t)$$

  where $\eta$ is the learning rate and the gradient is computed on just one sample $(x_t, y_t)$.

- **Naïve Bayes**: it just counts occurrences. You can keep running counts and update them as new data arrives. No need to re-scan old data.

### The `partial_fit()` API

Scikit-learn provides `partial_fit()` on models that support online learning. The key difference from `fit()`:

| `fit()` | `partial_fit()` |
|---------|----------------|
| Replaces any previous learning | Builds on previous learning |
| Expects all data at once | Accepts a chunk at a time |
| Can call repeatedly (starts fresh) | Must call repeatedly (accumulates) |

**Important first-call requirement:** On the first call to `partial_fit()`, you must pass `classes=[0, 1]` so the model knows all possible target values upfront. Otherwise it might see only class 0 in the first chunk and not know class 1 exists.

### Mini-Batch Size

You choose how many rows to process at a time:

- **1 row** (true online/stochastic): noisiest updates, slowest in Python due to loop overhead
- **1,000–100,000 rows** (mini-batch): good balance of speed and learning quality
- **All rows** (batch): defeats the purpose here

Larger mini-batches are faster in practice (vectorized operations) but use more memory. A common choice for this dataset would be something like **10,000–50,000 rows per chunk**.

### The Data Pipeline Pattern

The general pattern you'll use:

```
open file → read chunk → encode features → partial_fit → repeat → done
```

Using `pd.read_csv(..., chunksize=N)` gives you an iterator that yields DataFrames of N rows each. You never hold the full file in memory — just one chunk at a time.

### Single Pass vs. Multiple Passes

- **Single pass**: read the file once, front to back. Simplest. Works well with enough data (40M rows is usually plenty).
- **Multiple passes**: read the file several times. Can improve convergence for SGD but means re-reading 6+ GB from disk each time.

For this assignment, a single pass is likely sufficient and is the most "truly online" approach.

### Key Implications for This Assignment

1. **Feature encoding must also be online-compatible.** You can't use `OneHotEncoder` (needs to know all categories upfront). Use `FeatureHasher` instead — it works without knowing the full category set.
2. **Evaluation must be efficient.** You can compute log loss incrementally with a running sum — no need to store all predictions.
3. **The chronological ordering matters.** Training on earlier data and evaluating on later data mimics how the model would work in production.