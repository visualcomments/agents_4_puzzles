# Guarded sweep run_log list normalization fix

Fixed `megaminx_guarded_sweep.py` crash:

```
AttributeError: 'list' object has no attribute 'get'
```

Root cause: `pipeline_cli.py` can write `run_log.json` as a JSON list of report/event
objects, while the guarded sweep runner expected a single dict.

Changes:
- added `normalize_run_report(raw)`;
- added `load_run_report(path)`;
- added `report_path(report, key)`;
- normalized list-shaped run logs before reading `solver`, `stages`, and `kaggle_submit`;
- preserved Kaggle submit diagnostics such as `status=unknown` and public score.

This keeps the Kaggle CLI workflow unchanged: `kaggle competitions submit <COMPETITION> -f <FILE_NAME> -m <MESSAGE>`.
