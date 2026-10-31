# Reproducibility scope

The package supports three levels of verification:

1. **Open inspection and synthetic execution:** code/configuration inspection and unit/notebook checks without licensed content.
2. **Licensed regeneration:** full corpus construction and both article workflows when the user supplies lawful private inputs externally.
3. **Aggregate regression:** comparison of generated non-record-level values with frozen final references under declared exact/rounding tolerances.

The package does not redistribute WoS records, derived record tables, or unlicensed comparison series. Synthetic tests verify executable method invariants; they do not reproduce article numbers. Full numerical acceptance was completed privately on 30 October 2026 (563/563 checks matched); users can rerun it only when they have lawful access to equivalent private inputs.

A source-only staging candidate may be uploaded to a **private** GitHub repository. No public release, DOI, or immutable tag is created by this packaging review. See `docs/RELEASE_STATUS_AND_SCOPE.md`.
