# Optional chat-breakthrough artifacts

Place externally generated Megaminx submission CSVs here, or set `MEGAMINX_CHAT_ARTIFACTS=/path/to/csv_or_directory`.

The solver never trusts these files directly. It accepts only rows with `initial_state_id` and `path` columns, only official move names, and only paths that replay exactly from the bundled `test.csv` state to the central state and are strictly shorter than the incumbent bundled path.

This directory is intentionally empty in the portable baseline; it is an interface for TPU/NISS/rescue/history-beam outputs discussed in the chat export.
