# Megaminx GUI

Локальный запуск из корня репозитория:

```bash
pip install -r requirements-megaminx-gui.txt
python megaminx_gui_app.py
```

Ноутбук для Colab/Jupyter:

- `colab/agents_4_puzzles_megaminx_gui.ipynb`

GUI умеет:

- запускать `baseline_no_llm`, `best_tested_solver`, `optimized_assets_only`, `optimized_assets_v3_top150`, `optimized_assets_v3_top300`;
- запускать prompt-варианты `regular`, `improved`, `dataset_adapted`, `structured`, `heuristic_boosted`, `master_hybrid`;
- проверять все или выбранные `g4f`-модели;
- выбирать одну модель, набор моделей или автоматически использовать working-модели после probe;
- загружать `kaggle.json` и выполнять preflight / submit лучшего результата.
