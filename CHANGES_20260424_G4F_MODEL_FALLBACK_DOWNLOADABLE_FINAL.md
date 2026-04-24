# G4F model fallback fix - downloadable final

Compact rebuild from original archive; patched files copied from the validated v2 build.

- Fixed recovery crash when `baseline_code` is `None`.
- Treats g4f provider/auth/rate-limit/timeout/offline-baseline markers as failed model attempts.
- Continues to the next configured g4f model instead of stopping on baseline-generated CSV.
- Adds `csv_exists`, `model_failure_detected`, and `model_failure_markers` analytics fields.
