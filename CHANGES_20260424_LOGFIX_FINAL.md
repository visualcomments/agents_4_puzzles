# Megaminx notebook/repository logfix update — 2026-04-24

## Что было найдено в логах

- Ячейка Kaggle credentials падала с `ValueError: Kaggle credentials are placeholders...`, потому что многострочный placeholder обрабатывался как активный источник credentials.
- В предыдущих правках была возможна ошибка `NameError: name 'r' is not defined` при ручном редактировании raw-string поля.
- Один из промежуточных pipeline zip-файлов в рабочей директории оказался повреждённым как zip; новый архив пересобран с нуля из валидного базового архива.
- Аналитическая ячейка могла падать, если runner завершился раньше и не успел создать некоторые CSV/колонки.

## Что исправлено

- Добавлена устойчивая поддержка `KAGGLE_JSON_INLINE_ONE_LINE`.
- Поддержаны:
  - строка JSON в Colab-form;
  - строка JSON в Python-коде;
  - Python dict literal.
- Многострочный placeholder теперь безопасно пропускается.
- Полный Kaggle key не печатается в output, только masked preview.
- После ключевых ячеек выводятся последние 30 строк логов.
- Итоговый zip результатов пересоздаётся при каждом запуске.
- Аналитика self-improving prompts стала устойчивой к пустым/частичным данным.
