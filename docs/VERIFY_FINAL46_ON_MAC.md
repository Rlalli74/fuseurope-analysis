# One-command final numerical acceptance on the validated Mac

This reruns **only** aggregate assembly and validation over the existing private outputs; it does not rerun network computations, download or publish any licensed data. The previous private pipeline and its 561 scientific acceptance checks already passed; the public repository now uses only its current 46-column representation and its observed SHA-256.

1. Extract the new source ZIP into `~/FusEurope_VERIFICA_MAC_20261030/FusEurope_GitHub_FINAL46_20261030` (a **new** folder; do not merge into the old code checkout), so the source root is `.../FusEurope_GitHub_FINAL46_20261030/fuseurope-analysis/`.
2. Keep the old `RISERVATO_NON_PUBBLICARE` private working folder unchanged.
3. Copy `VERIFICA_PACCHETTO_FINALE46_MAC.command` next to the code folder (or run it from anywhere); open Terminal and run `bash /PATH/TO/VERIFICA_PACCHETTO_FINALE46_MAC.command`.
4. Read `~/FusEurope_VERIFICA_MAC_20261030/RISERVATO_NON_PUBBLICARE/results/final_acceptance_FINAL46/FINAL_ACCEPTANCE_REPORT.md`. Expected for the validated snapshot: 563/563 PASS. If not, do not adjust expected values to force success; inspect the failure fields.

The private data remain in the original private directory. No generated data or numerical reports are copied into the public source tree.
