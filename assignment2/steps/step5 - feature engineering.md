

## 5. Feature Engineering for Categorical Variables

### The Dataset's Features

The Avazu dataset columns are:

| Column | Description |
|--------|-------------|
| `id` | Unique impression ID (not a feature — just an identifier) |
| `click` | Target variable: 0 or 1 |
| `hour` | Timestamp in YYMMDDHH format (e.g., 14102100 = 2014-10-21 hour 00) |
| `C1` | Anonymized categorical |
| `banner_pos` | Ad position on the page |
| `site_id` | Hashed site identifier |
| `site_domain` | Hashed site domain |
| `site_category` | Hashed site category |
| `app_id` | Hashed app identifier |
| `app_domain` | Hashed app domain |
| `app_category` | Hashed app category |
| `device_id` | Hashed device identifier |
| `device_ip` | Hashed device IP |
| `device_model` | Hashed device model |
| `device_type` | Device type |
| `device_conn_type` | Connection type |
| `C14`–`C21` | More anonymized categoricals |

**Critical point from the instructions:** ALL of these are **nominal categorical** variables, even the ones that look numeric. `banner_pos=0` vs `banner_pos=1` doesn't mean position 1 is "greater" than position 0 in a meaningful numeric sense. They're just labels.

### The Encoding Problem

Machine learning models need numeric input. You can't feed the string `"a]b3d9f2"` to logistic regression. You need to convert categories to numbers — but **how** matters enormously.

**Wrong approach — label encoding:** Assigning `site_id="abc" → 1`, `site_id="def" → 2`, etc. This implies an ordering (2 > 1) that doesn't exist. The model would treat these as numeric values and compute meaningless gradients.

**Correct approach — one-hot encoding:** Create a binary column for each unique value.

| site_id | → | site_id=abc | site_id=def | site_id=ghi |
|---------|---|-------------|-------------|-------------|
| abc | | 1 | 0 | 0 |
| def | | 0 | 1 | 0 |
| ghi | | 0 | 0 | 1 |

**The problem with one-hot encoding here:** `device_ip` alone has hundreds of thousands of unique values. `device_id` has millions. One-hot encoding all features would create a feature matrix with millions of columns — impossible to hold in memory. And in online learning, you don't know all possible values upfront.

### The Feature Hashing Trick

**Feature hashing** solves both problems. Instead of creating one column per category value, you use a hash function to map each value to one of $n$ fixed buckets:

$$\text{feature\_index} = \text{hash}(\text{"site\_id=abc"}) \mod n$$

With $n = 2^{18} = 262{,}144$ (for example), every possible category value maps to one of 262,144 positions. The output is a sparse vector of fixed, known size — regardless of how many unique values exist.

**Advantages:**
- Fixed memory regardless of cardinality
- Works with never-before-seen values (no "unknown category" problem)
- No need to store a mapping dictionary
- Naturally produces sparse vectors (most entries are 0)

**Disadvantage — collisions:** Different values can hash to the same bucket. `"site_id=abc"` and `"device_model=xyz"` might both map to bucket 7,412. This adds noise but rarely hurts much in practice, especially with a large enough $n$.

### `FeatureHasher` in Scikit-learn

```python
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=2**18, input_type='dict')
```

It expects input as a list of dictionaries, where each dictionary represents one row:

```python
# One impression's features
{"site_id": "abc", "banner_pos": "0", "device_type": "1", ...}
```

Key parameters:
- **`n_features`**: number of hash buckets. Larger = fewer collisions but more memory. Powers of 2 are conventional. Common choices: $2^{16}$ (65K) to $2^{20}$ (1M).
- **`input_type='dict'`**: expects `[{feature: value}, ...]`
- **`non_negative=True`**: **set this if using MultinomialNB**, which requires non-negative inputs. Default `FeatureHasher` uses a signed hash trick (values can be +1 or -1) which is fine for SGD but breaks MultinomialNB.

### Preparing Features for Hashing

You need to convert each row of the DataFrame to a dictionary with **string values** (so the hasher treats them as categories, not numbers):

```python
# Convert selected columns to feature dictionaries
feature_cols = ['banner_pos', 'site_id', 'site_category', ...]
dicts = chunk[feature_cols].astype(str).to_dict(orient='records')
X = hasher.transform(dicts)
```

The output `X` is a **sparse matrix** (CSR format) — memory-efficient because most values are zero.

### Which Features to Include?

Not all columns are equally useful. You need to think about:

**1. Cardinality (number of unique values)**

| Feature | Approx. Unique Values | Concern |
|---------|----------------------|---------|
| `banner_pos` | ~7 | Fine — low cardinality |
| `site_category` | ~26 | Fine |
| `device_type` | ~5 | Fine |
| `device_ip` | ~6M | Very high — mostly unique, low predictive value per value |
| `device_id` | ~3M | Very high — same concern |

Very high-cardinality features like `device_ip` and `device_id` have so many unique values that most appear only once. A single occurrence gives no generalizable signal and just adds collision noise in the hash space. You may want to **exclude** them.

**2. Mutual exclusivity**

`site_*` columns and `app_*` columns are **mutually exclusive** — each impression is either on a website or in an app, not both. When `site_id` has a value, `app_id` is typically a placeholder (and vice versa). This isn't a problem for the model but is worth noting in your EDA.

**3. The `hour` column**

`hour` is in YYMMDDHH format (e.g., `14102115`). You can extract useful sub-features:
- **Hour of day** (0–23): captures daily patterns (e.g., higher CTR in evenings)
- **Day of week**: captures weekly patterns

Using the raw `hour` value as a category would give ~240 unique values (10 days × 24 hours), each appearing in a limited time window — not ideal for generalization.

**4. Feature selection approach**

For the report, reasonable strategies include:
- Start with all features, measure baseline log loss
- Remove very high-cardinality features (`device_ip`, `device_id`), measure again
- Extract time features from `hour`, measure again
- Compare and justify your final selection

### The `n_features` Trade-off

| `n_features` | Collisions | Memory | Speed |
|--------------|-----------|--------|-------|
| $2^{14}$ (16K) | Many | Low | Fast |
| $2^{18}$ (256K) | Few | Moderate | Moderate |
| $2^{22}$ (4M) | Very few | High | Slower |

A good starting point is $2^{18}$. You can justify your choice by testing a couple of values and comparing log loss — if increasing $n$ doesn't improve log loss, the smaller value is sufficient.

### Key Takeaways

1. **All features are categorical** — never treat them as numeric
2. **Feature hashing** is the only practical encoding for online learning with high-cardinality features
3. **Drop `id`** (it's an identifier, not a feature)
4. **Consider dropping `device_ip` and `device_id`** (too many unique values)
5. **Engineer time features** from the `hour` column
6. **Use `non_negative=True`** in `FeatureHasher` if you're using `MultinomialNB`
7. The choice of features and `n_features` are decisions you need to justify in the report

# Questions
/// Can't we add geography for ip?