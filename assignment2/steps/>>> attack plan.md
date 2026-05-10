# Plan: CTR Prediction Notebook (Updated)

**Decisions locked in:**
- **New notebook** at `assignment2/notebook.ipynb` — existing `input/Assignment 2 - notebook.ipynb` left untouched
- **Data path** — `../data/train.gz` (relative from the notebook location) — file already present, no download step needed

---

### Notebook Sections

**Section 0 — Setup**
- All imports, two named constants at the top:
  - `DATA_PATH = "../data/train.gz"`
  - `CHUNK_SIZE = 50_000`
  - `N_FEATURES = 2**18`

**Section 1 — Exploratory Data Analysis**
- Load first 100K rows via `pd.read_csv(DATA_PATH, nrows=100_000)`
- Show shape, dtypes, sample rows, missing values
- Bar chart: click distribution (class imbalance)
- Table: unique value counts per column → motivates which to drop
- Line chart: average CTR by hour-of-day → motivates time feature extraction

**Section 2 — Feature Selection**
- Markdown-only cell: explicit inclusion/exclusion table with rationale
- Drop: `id`, `device_ip`, `device_id`
- Transform: `hour` → extract `hour_of_day` (mod 100) and `day_of_week`
- Keep everything else as categorical strings

**Section 3 — Feature Encoding**
- Plain-English explanation of FeatureHasher
- Define `FEATURE_COLS` list
- Two hasher instances: `hasher_sgd` (default) and `hasher_nb` (`non_negative=True`)
- Helper function: `encode_chunk(chunk, hasher)` → sparse matrix + y

**Section 4 — Baseline**
- Single pass computing total clicks / total rows = average CTR
- Apply that constant to test rows → baseline log loss
- This is the "do-nothing" benchmark

**Section 5 — Train/Test Split**
- Markdown cell with diagram explaining temporal vs. random split
- `TRAIN_CUTOFF_CHUNK = 640` (rows 0–32M train, 32M–40M test at chunksize=50K)

**Section 6 — SGD Logistic Regression**
- `SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, learning_rate='optimal')`
- Loop with `tqdm` progress bar, timer, incremental log loss on test chunks

**Section 7 — Naïve Bayes**
- `MultinomialNB(alpha=1.0)`
- Identical loop structure, timer, incremental log loss

**Section 8 — Results & Discussion**
- Summary table: Baseline / SGD / NB → log loss + training time
- One markdown cell per report question from the instructions

---

**Relevant files**
- `assignment2/notebook.ipynb` — to be created
- train.gz — source data, already present
- `assignment2/instructions.md` — 7 report questions map to Section 8

**Verification**
1. Section 1 runs in < 10 seconds on 100K rows
2. Both model loops complete in a single pass without memory errors
3. Both models beat baseline log loss
4. NB training time < SGD training time


# Comparison with class materials

Let me read the notebook more carefully alongside the lecture material. 

Read [](file:///Users/martonhorvath/all_repos/abs-dsmarketing/assignment2/notebook.ipynb#1-1), lines 1 to 100

Here's a structured compliance check against the lecture (Slides 1–55):

---

## Compliance Assessment

### What aligns well

| Aspect | Lecture (Slide) | Notebook |
|--------|----------------|---------|
| Problem framing | Binary classification, predict click *probability* (Slides 11–13) | Correctly framed |
| `predict_proba()` | Slide 13 | Used correctly |
| `SGDClassifier` + `partial_fit()` | Slide 24 | Used correctly |
| `MultinomialNB` + `partial_fit()` | Slide 23 (online learning) | Used correctly |
| Log loss formula $-[y\ln p + (1-y)\ln(1-p)]$ | Slide 19 | `incremental_log_loss()` implements this exactly |
| Class imbalance discussion | Slides 14–18 | Covered in EDA and discussion prompts |
| Temporal/sequential split logic | Slide 23 (observations come in continuously) | Temporal split with 80/20 |
| Naïve Bayes = counting approach | Slides 27–33 | `MultinomialNB` does this |
| Conditional independence assumption | Slide 29 | Mentioned in Section 7 discussion |
| Laplace smoothing | Implied by Slide 33 (unseen values) | `alpha=1.0` in NB |

---

### Compliance gaps

**1. Feature encoding: `FeatureHasher` vs `OneHotEncoder`**

The lecture explicitly shows `OneHotEncoder` (Slides 36–37):
```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train)
```
The notebook uses `FeatureHasher`, which is more practical but **not covered in the lecture at all**. This is the biggest drift. If a grader looks for the course-taught encoding approach, this will stand out.

**2. SGD hyperparameters are not anchored to the lecture**

The lecture shows `SGDClassifier()` with no parameters (Slide 24). The notebook uses:
```python
SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, learning_rate="optimal")
```
None of these choices are explained relative to course material. There's even a `# IMPROVE` comment in that cell acknowledging this. The lecture discusses the learning rate schedule $\eta_t = 1/\sqrt{t}$ (Slide 43) — `learning_rate="optimal"` uses sklearn's own formula, not the one taught.

**3. `alternate_sign=False` detail for NB**

This is a technical implementation detail needed because `MultinomialNB` requires non-negative inputs. It's not derivable from the lecture — it's a scikit-learn internals concern. Fine as a practical necessity, but it should be briefly explained as such, not presented as a course concept.

**4. Mini-batch size of 50,000**

Slide 50 introduces mini-batches conceptually, but the lecture gives no guidance on choosing a size. `CHUNK_SIZE = 50_000` is reasonable but unexplained.

**5. `classes=[0, 1]` in `partial_fit()`**

Not mentioned in the lecture's code snippet (Slide 24). It's required by sklearn on the first call, but a student unfamiliar with the API wouldn't know why it's there. Worth a one-line comment.

---

### Summary by risk level

| Gap | Risk |
|-----|------|
| `FeatureHasher` instead of `OneHotEncoder` | High — lecture explicitly teaches OHE |
| SGD parameters unexplained vs lecture | Medium — the `# IMPROVE` comment already flags this |
| `alternate_sign=False` has no explanation | Low — practical necessity |
| Learning rate `"optimal"` vs lecture's $1/\sqrt{t}$ | Medium — lecture teaches the formula; notebook silently uses a different one |

The most defensible fix is to **add a brief note in Section 3** explaining why you deviate from `OneHotEncoder` (it can't handle unseen categories in online learning, and the dataset is too large to fit all unique values upfront), and to **tie the SGD parameter choices back to the lecture concepts** (log loss from Slide 42, learning rate from Slide 43).