# Aethel Learnings

- Training stack
  - Keep Modal args direct (no extra `--`); expose debug flag via entrypoints.
  - Use packaged datasets only; avoid script-based splits. Filter empty/whitespace rows to prevent all-pad batches.
  - Add split fallbacks and correct STSB split (`dev`) for mini-MTEB eval.

- Model stability
  - Avoid all-`-inf` attention rows; zero them before softmax to stop NaNs.
  - If pooled+memory collapses, inject tiny noise to break symmetry.
  - Fallback to unmasked mean when mask sum is zero; clamp denominators.
  - Slightly scale token embeddings and head projections at init to avoid dead embeddings.

- Distillation/debug
  - Truncate teacher inputs to their max positions (BGE 512) to prevent shape errors.
  - Align teacher hidden dims before stacking/averaging; align sparse vocab dims too.
  - InfoNCE and cosine align to min hidden dim; guard temps and NaNs with `nan_to_num`.
  - Debug context: report student/teacher norms, mask sums, raw NaN flags to catch collapse early.

- Loss tuning
  - Lowered InfoNCE/matryoshka weights (0.25 each) to reduce early instability.
  - Sanitized losses with `nan_to_num` (no forced zeros of finite values).

- Dataset choices (current defaults)
  - `wikitext-103-raw-v1` train[:20000], `ag_news` train[:20000], `amazon_polarity` train[:20000].

- Known pitfalls/avoided
  - Percent split strings unsupported in current datasets build; use absolute ranges.
  - BookCorpusOpen, CodeSearchNet python (script datasets) blocked; replaced with packaged sets.
  - All-pad masks and all-`-inf` attention led to student NaNs/zeros; fixed via filtering and attention guard.

- Current status
  - Student embeddings now finite and non-zero; losses decrease after step 0. Ready for full Modal run without debug.


The student embeddings were collapsing because attention rows were all -inf after masking, which made softmax produce NaNs; we then nan_to_num’d them to zero, so pooled outputs went to zero and stayed there. Fixes that solved it:

In attention.py, after masking, any all--inf score row is set to 0.0 before softmax, preventing NaNs.
In aethel_model.pooled_mean, we fallback to unmasked mean if mask sum is zero, so pooled vectors aren’t zeroed by empty masks.
We added tiny noise if pooled+memory is zero, and scaled up embedding/head init to avoid dead starts.
Dataset filtering now removes empty/whitespace texts to avoid all-pad batches.
After these, student_dense_norm became finite and losses started changing.

