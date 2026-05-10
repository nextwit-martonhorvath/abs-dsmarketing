# EDA

We can check outliers in the features we want to keep

# Feature encoding

Should we add day in addition to hour

# Use of one-hot-encoder instead of hashing:

OHE is conceptually simpler (explicit 1-of-K encoding, no hash collisions, no sign trick), and since we already have the random sample in memory we can fit the encoder there. One encoder handles both models because OHE always produces 0/1 values (non-negative), eliminating the need for two separate hashers.

## Steps

Imports & constants — swap FeatureHasher for OneHotEncoder from sklearn.preprocessing; remove the N_FEATURES = 2**18 constant (no longer needed)

Section 3 markdown — rewrite the encoding explanation: OHE maps each "column=value" to its own binary column; handle_unknown='ignore' silently drops values not seen during fit (equivalent to all-zeros, similar to a hash miss)

Feature encoding cell — remove hasher_sgd / hasher_nb; add a single encoder fitted on the sample:

Rewrite encode_chunk(chunk, hasher) → encode_chunk(chunk) using encoder.transform(dicts_df) instead of hashing

SGD training cell — update call: encode_chunk(chunk, hasher_sgd) → encode_chunk(chunk); remove the '''IMPROVE...''' block

NB training cell — same call update

## Relevant files

notebook.ipynb — cells: imports, Section 3 markdown, feature encoding, SGD training, NB training

## Verification

Run the sanity-check cell — output shape should be (5, N) where N = total unique values across selected feature columns in the sample
Confirm NB cell runs without error (OHE output is always ≥ 0, so no alternate_sign issue)
Final log loss values should be in a similar ballpark to the hashing approach (may differ slightly due to no collisions)

## Decisions

Encoder is fit on sample (~100K rows across all 10 days) — rare values in the full dataset get handle_unknown='ignore' (all-zeros row), which is the correct behaviour
hour_of_day must be added to the sample before fitting (same % 100 transform)
OHE matrix width ≈ sum of unique values per column in the sample; with moderate-cardinality features this is manageable as a sparse matrix
sparse_output=True keeps memory usage comparable to hashing

## Further Considerations

Feature matrix width: If site_id or similar has thousands of unique values in the sample, the OHE matrix could be very wide (potentially 50K+ columns). This is fine for SGD but may slow down NB. Do you want to cap cardinality (e.g. drop columns with >500 unique values in sample) or proceed as-is?