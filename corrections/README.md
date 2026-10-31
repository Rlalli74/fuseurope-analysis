# Manual corrections

Manual decisions are stored as data rather than hidden in notebooks or source code.

- `author_identity_overrides.csv` contains curated `MERGE` and `KEEP_SEPARATE` decisions applied after automatic author clustering. The final file incorporates the completed transliteration/orthographic and targeted historical-snapshot identity audit, including the Artsimovich correction.
- `author_identity_unresolved_priority.csv` is a review aid, not an instruction to merge all listed pairs.

Record-level evidence from licensed Web of Science records is retained separately in the restricted reproducibility archive and is not part of a public release. See `AUTHOR_OVERRIDE_GUIDE.md` and `../docs/AUTHOR_IDENTITY_TRANSLITERATION_AUDIT_PUBLIC_SUMMARY.md`.
