Read [](file:///Users/martonhorvath/all_repos/abs-dsmarketing/data)

Updated memory file [](file:///Users/martonhorvath/Library/Application%20Support/Code/User/workspaceStorage/bb4029242be636434a89c1834c21e59c/GitHub.copilot-chat/memory-tool/memories/ODA5YTA3NzItNDlhNy00YzcxLTgwYmMtYjg4MzM4MmIzZWVi/plan.md)

Plan updated. Here's the revised version:

---

## Plan: CTR Prediction Notebook (Updated)

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