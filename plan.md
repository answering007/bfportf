# План развития проекта bfportf

> Рабочая ветка: **dev**
> Базовый коммит: `314397c` ("set train_weights to out-of-sample results")
> Текущая версия пакета: `0.1.3` (`src/bfportf/__about__.py`)
> Документ ведётся итеративно: после каждого завершённого шага обновляются разделы «Статус» и «Журнал изменений».

## Легенда статусов

- `[x]` — сделано
- `[~]` — в работе
- `[ ]` — не начато
- `[-]` — отменено / отложено (с указанием причины)

## Как работаем

1. Работа идёт **поэтапно**: этапы 0 → 1 → 2 → 3, внутри этапа — по пунктам чек-листа.
2. Каждый этап завершается зелёными проверками (см. «Критерии приёмки» этапа) и обновлением этого файла.
3. Один этап = один атомарный коммит (или серия логических коммитов) в ветке `dev`.
4. Перед началом этапа пункт помечается `[~]`, после проверки — `[x]`.
5. Любое отклонение от плана фиксируется в разделе «Журнал изменений» с причиной.
6. Все команды разработчика запускаются через `uv run ...` (ADR-2); `hatch` в проекте не используется, `hatchling` — только build-backend.

## Оглавление

- [Статус](#статус)
- [Что уже сделано](#что-уже-сделано)
- [Принятые решения (ADR)](#принятые-решения-adr)
- [Проверенные факты окружения](#проверенные-факты-окружения)
- [Реестр дефектов](#реестр-дефектов)
- [Этап 0 — блокеры](#этап-0--блокеры-012)
- [Этап 1 — корректность расчётов](#этап-1--корректность-расчётов-013)
- [Этап 2 — архитектура](#этап-2--архитектура-020)
- [Этап 3 — упаковка и документация](#этап-3--упаковка-и-документация-02x)
- [Тестовый набор](#тестовый-набор-целевое-состояние)
- [Журнал изменений](#журнал-изменений)

---

## Статус

| Этап | Содержание | Статус |
|---|---|---|
| Аудит | Ревизия кода, воспроизведение дефектов на живом окружении | `[x]` |
| Решения | Зафиксированы ADR-1 (breaking changes в 0.2.0) и ADR-2 (единый тулчейн — `uv`) | `[x]` |
| 0 | Блокеры P0: работоспособность на актуальных numpy/pandas/scipy, чистый линт | `[x]` (0.1.2) |
| 1 | Корректность расчётов: CAGR, NaN/inf, tolerance, семантика весов, валидация | `[x]` (0.1.3) |
| 2 | Архитектура: разбиение на слои, реестр метрик, стратегии поиска, публичный API | `[x]` (0.2.0) |
| 3 | Упаковка, CI, документация, данные примеров | `[x]` (0.2.0) |

**Готовность к использованию:** все этапы 0-3 закрыты. Публичный API, три стратегии, реестр метрик с направлением, векторный fast-path (2.7x), 124 теста и покрытие 96%, `mypy` без ошибок, CI-workflow и pre-commit, wheel 28 КБ (с `py.typed`) и sdist 33 КБ без `prices.csv`/`uv.lock`. Релиза в PyPI нет по ADR-5 — артефакты прикладываются к GitHub-релизу.

---

## Что уже сделано

### Аудит (этап предварительного анализа)

- `[x]` Прочитан весь исходный код: `utils.py`, `targets.py`, `optimization.py`, `report.py`, `__init__.py`, `__about__.py`.
- `[x]` Проанализированы `pyproject.toml`, `uv.lock`, `.gitignore`, `LICENSE.txt`, `README.md`.
- `[x]` Разобраны оба примера-ноутбука (`examples/BfPortf 01 Simple example.ipynb`, `02 Custom target function.ipynb`) — восстановлены реальные сценарии вызова API.
- `[x]` Проверена история git (`master`, 7 коммитов) — CI-workflow был удалён ("Remove test workflow"), `.github` отсутствует.
- `[x]` Дефекты **воспроизведены** на живом окружении (см. ниже), а не только вычитаны из кода.
- `[x]` Сформулированы рекомендации по архитектуре и приоритизированный план (этот документ).

### Этап 0 (0.1.2) — выполнено

- `[x]` `np.in1d` -> `np.isin` в `optimization.py` (P0-1).
- `[x]` `scipy>=1.8` в `dependencies`; нижние границы `numpy>=1.22`, `pandas>=1.5`, `tqdm>=4.64` (P0-2).
- `[x]` `create_report`: `astype(object)` перед применением `custom_formatting` (P0-3).
- `[x]` Конфиг ruff -> `[tool.ruff.lint]`; `black` заменён на `ruff format` (P0-4, P0-5).
- `[x]` `requires-python = >=3.9`, классификаторы 3.9-3.13, hatch-окружения удалены, добавлен `[dependency-groups] dev` (P0-6, ADR-2).
- `[x]` Смоук-тесты `tests/test_smoke.py` (4 теста), версия `0.1.2`.
- `[x]` `uv lock` + `uv sync --group dev`; `uv.lock` актуализирован.

Проверки этапа 0: `uv run pytest -q` -> 4 passed; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 10 files already formatted.

### Этап 1 (0.1.3) — выполнено

- `[x]` **P1-1.** CAGR: `periods = len(returns) / days_per_year` (проверено: 1%/день x 252 -> 11.274%).
- `[x]` **P1-2.** Политика `NaN`/`inf`: метрики возвращают `np.nan` вместо `inf`; `np.isfinite`-фильтр кандидатов в `find_next_symbol`; добавлен `validate_returns()` (NaN, inf, дубли колонок/индекса, немонотонный индекс; константные колонки -> warning).
- `[x]` **P1-3.** Параметр `tolerance` в `find_portfolios` / `find_portfolio_for_symbol` / `find_next_symbol`.
- `[x]` **P1-4.** `get_portfolio_return(..., normalize="sum"|"none")`, семантика весов документирована (доли аллокации, нормализация к 1 = неявное плечо).
- `[x]` **P1-5.** Валидация символов и весов -> `UnknownSymbolError` / `DataValidationError` (новый модуль `exceptions.py`).
- `[x]` **P1-6.** Единый контракт метрик: все получают `np.ndarray`; отчёт собирается без `agg`-хака.
- `[x]` **P1-7.** Согласованы `ddof=1`, добавлен `risk_free`; `recovery_factor = pnl_target / |max_drawdown|`.
- `[x]` **P1-8.** `calculate_metric` переиспользует `get_portfolio_return`, принимает `Series` / `ndarray` / `None`.
- `[x]` **P1-9.** Docstring и аннотации приведены в соответствие с фактическим поведением.
- `[x]` **P1-10.** `groupby(..., dropna=False)`; `NaN`-значения уходят в конец сортировки.
- `[x]` **P1-11.** Параметр `benchmark_weights` в `create_report` (дефолт сохраняет поведение 0.1.x).
- `[x]` Тесты: `conftest.py`, `test_utils.py`, `test_utils_portfolio.py`, `test_targets.py`, `test_optimization.py`, `test_report.py`, `pytest-cov` в dev-группе.

Проверки этапа 1: `uv run pytest -q --cov=bfportf` -> 85 passed, покрытие 95%; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 17 files already formatted; `uv run mypy src` -> те же 6 `import-untyped` (baseline этапа 0, новых нет).

### Этап 2 (0.2.0) — выполнено

- `[x]` Слои: `data.py`, `scaling.py`, `portfolio.py`, `metrics/` (`base`, `basic`, `registry`), `search/` (`base`, `greedy`, `beam`, `exhaustive`); `utils.py` — только `return_to_price` / `price_to_return` / `drawdown_pct`.
- `[x]` `Metric(name, func, maximize, vectorized)` + `METRICS` (10 метрик) + `resolve_metric("sharpe" | Metric | callable)`.
- `[x]` **ADR-3.** `GreedySearch` (дефолт), `BeamSearch(width=5)`, `ExhaustiveSearch(max_combinations=2_000_000)`, протокол `SearchStrategy` и `SearchConfig`.
- `[x]` **ADR-4.** `find_portfolios` -> `list[PortfolioResult]`; `results_to_frame()`; `PortfolioResult.to_dict()/to_frame()`.
- `[x]` Публичный API: `__init__.py` (`__all__` — 35 имён), `py.typed`, `__version__`; `bfportf.targets` и `bfportf.optimization` сохранены как фасады.
- `[x]` **P2-8.** Векторный fast-path: 200 x 1000, `max_number=3` — **0.6 c против 1.6 c** (2.7x) при идентичном результате.
- `[x]` **P2-7.** `pandas-stubs` / `types-tqdm`, `[tool.mypy]` с `disallow_untyped_defs` — `uv run mypy src` -> **0 ошибок** (было 6).
- `[x]` Тесты: `test_search.py`, `test_strategies.py` (exhaustive = brute force, beam >= greedy, `width=1` = greedy), `test_metrics_registry.py` (vectorized = scalar), `test_api.py`, `test_smoke.py`; слои — `test_data_scaling.py`, `test_portfolio.py`, `test_utils.py`.
- `[x]` Примеры перегенерированы под новый API и очищены от output-ов (NB02: 1 МБ -> 5 КБ); README переписан; добавлен `CHANGELOG.md`.

Проверки этапа 2: `uv run pytest -q --cov=bfportf` -> 124 passed, покрытие 96%; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 34 files already formatted; `uv run mypy src` -> Success, 0 issues.

### Этап 3 (0.2.0) — выполнено

- `[x]` **P2-2.** `.github/workflows/ci.yml`: job `quality` (matrix 3.10-3.13, `uv sync --locked`, ruff, ruff format, mypy, pytest с покрытием), job `build` (wheel/sdist, проверка содержимого, установка в чистое окружение), job `examples` (резолв extra).
- `[x]` **P2-2.** `.pre-commit-config.yaml`: `uv-lock`, `ruff`, `ruff-format`, `nbstripout`, whitespace/EOF, `check-added-large-files` (1 МБ).
- `[x]` **P2-3.** Метаданные: реальные URL GitHub, автор, `license-files` (PEP 639), `hatchling>=1.27`, `keywords`, описание; заголовки SPDX в `__about__.py` и `tests/__init__.py`.
- `[x]` **P2-4.** `examples/prices.csv` (14.5 МБ) убран из индекса git и из sdist; ноутбуки качают данные по URL; output-ы ноутбуков вычищены.
- `[x]` **P2-5.** `[project.optional-dependencies] examples` (`scikit-learn`, `quantstats`, `jupyter`); `uv.lock` пересобран и включает extra.
- `[x]` **P2-9.** README: установка из wheel (ADR-5), Quickstart, метрики, стратегии, веса, команды разработки; `CHANGELOG.md` с разделом `[Unreleased]`.
- `[x]` **P2-4.** Примеры переведены на API 0.2: слайс вместо `train_test_split`, inverse-vol веса оцениваются только in-sample (тест масштабируется теми же весами), кастомная метрика через `Metric`, стратегии сравниваются на одной вселенной, показаны `tolerance`, ошибки и инверсные ноги. Добавлен ноутбук `03 Metrics and strategies`.
- `[x]` **P2-4.** Данные примеров: `assets/prices.parquet` (4.5 МБ, 500 символов, `float32`) вместо `examples/prices.csv` (13.9 МБ); файл не в git (`/assets/*.parquet`), исключён из sdist, публикуется ассетом GitHub-релиза. Ноутбуки берут локальную копию (`assets/` или рядом), иначе скачивают ассет `releases/latest/download/prices.parquet`, и только как последний резерв конвертируют CSV из закреплённого коммита `314397c`. Чтение parquet требует `pyarrow` (в `dependencies`).
- `[x]` **ADR-5.** PyPI-бейджи удалены, установка документирована как `pip install <wheel>`, `uv build` входит в CI.
- `[x]` Проверка сборки: `uv build` -> wheel 28 КБ (24 файла, `py.typed`, без CSV) и sdist 33 КБ (без `prices.csv`, `uv.lock`, `plan.md`); установка wheel в чистое venv -> `0.2.0 35 sharpe`.
- `[x]` YAML-файлы (CI и pre-commit) провалидированы локально (`yaml.safe_load`).
- `[-]` **P2-10.** Линт `.ipynb` остаётся исключённым осознанно: в ячейках примеров порядок импортов и длинные URL — `extend-exclude = ["examples"]` сохранён.

Проверки этапа 3: `uv run pytest -q` -> 124 passed; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 34 files already formatted; `uv run mypy src` -> Success; `uv lock --check` -> lock актуален; `uv build` + установка wheel в чистом venv -> OK.

---

## Принятые решения (ADR)

| ID | Дата | Решение | Обоснование и следствия |
|---|---|---|---|
| ADR-1 | 2026-11-04 | **Обратная совместимость API 0.1.x не поддерживается.** Breaking changes допускаются в минорном релизе 0.2.0 (semver для 0.x: минор = breaking) | Не тратим силы на deprecation-шимы и дубли сигнатур. Вместо шимов: раздел `Breaking changes` в `CHANGELOG.md`, обновлённые ноутбуки-примеры и README. Патч-релизы 0.1.2 / 0.1.3 остаются совместимыми (только исправления) |
| ADR-2 | 2026-11-04 | **Единый тулчейн — `uv`.** Зависимости и dev-группы в `pyproject.toml` (`dependencies`, `[dependency-groups]`, `[project.optional-dependencies]`); воспроизводимость — `uv lock` + `uv.lock` в git; запуск задач — `uv run ...` | `hatchling` остаётся **только** build-backend в `[build-system]`. Удаляются `[tool.hatch.envs.*]`, `[[tool.hatch.envs.all.matrix]]` и hatch-скрипты; их заменяют `uv run pytest`, `uv run ruff check .`, `uv run black --check .`, `uv run mypy src`. Локальный `.venv` уже создан `uv 0.12.13`, `hatch` в PATH отсутствует. CI — `uv sync --locked` |
| ADR-3 | 2026-11-04 | **Стратегии поиска публичны.** `GreedySearch` (по умолчанию), `BeamSearch(width=5)`, `ExhaustiveSearch(max_combinations=2_000_000)` — классы за протоколом `SearchStrategy`; `find_portfolios(..., strategy=...)` | Разные компромиссы «качество/время»: greedy — быстро и детерминированно, beam — обходит часть локальных оптимумов за `B`-кратное время, exhaustive — гарантированный оптимум на маленьких вселенных (n <= 15) и эталон для тестов. `max_combinations` защищает от случайного запуска на 700 бумагах (~57 млн сочетаний). `find_next_symbol` / `find_portfolio_for_symbol` остаются внутренними |
| ADR-4 | 2026-11-04 | **Результат поиска — `list[PortfolioResult]`.** `PortfolioResult(symbols: tuple[str, ...], weights: pd.Series, value: float, metric_name: str)` + `results_to_frame()`; `maximize=None` берётся из `Metric` | Убирает `np.ndarray` в object-колонке DataFrame, даёт типизацию и веса портфеля. Breaking change разрешён ADR-1; примеры и README обновляются, табличное представление сохраняется через `results_to_frame` |
| ADR-5 | 2026-11-04 | **Публикации в PyPI не будет.** Распространение — `wheel`/`sdist`, прикладываемые к релизу на GitHub; бейджи PyPI из README удаляются, установка документируется как `pip install <путь к wheel>` | Решение пользователя. Следствия: `uv build` входит в CI, версия остаётся в `0.x`, `[project.urls]` ведут на GitHub, `CHANGELOG.md` ведётся как история проекта, а не как источник для PyPI |

Следствия по этапам:

- **Этап 0:** вместо правки hatch-матрицы — удаление hatch-окружений и скриптов; `uv lock` для актуализации lock-файла.
- **Этап 2:** пункт про deprecation-шимы отменяется (ADR-1) — вместо него `Breaking changes` в CHANGELOG + обновление примеров.
- **Этап 3:** `[dependency-groups] dev`, `uv sync --locked` в CI, pre-commit через `uv run`.
- **Этап 2 (ADR-3):** три стратегии в `search/`, `greedy` по умолчанию, `exhaustive` дополнительно как тест-эталон.
- **Этап 2 (ADR-4):** `find_portfolios` -> `list[PortfolioResult]`; примеры и README переводятся на `found[0].symbols`.

---

## Проверенные факты окружения

| Компонент | Версия / факт |
|---|---|
| Python (venv `.venv`) | 3.14.7 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| tqdm | 4.70.1 |
| scipy | 1.18.1 — установлен и объявлен в `dependencies` (этап 0) |
| `np.in1d` | отсутствует в NumPy 2.x; в коде заменён на `np.isin` (P0-1) |
| `.github` | отсутствует |
| `uv` | 0.12.13 (`C:/Users/Pavel/.local/bin/uv.exe`); локальный `.venv` создан `uv` |
| `hatch` | в PATH отсутствует (не установлен) — учтено в ADR-2 |
| `tests/` | 124 теста (этап 2), покрытие `src/bfportf` — 96% |
| `examples/prices.csv` | 14.5 МБ, в индексе git |
| ruff | 0.17.3 (`uv run ruff`) — линт и `ruff format` |
| pytest | 9.2.0 + pytest-cov 7.1.0; `uv run pytest -q --cov=bfportf` -> 124 passed, 96% |
| mypy | 2.4.1 + `pandas-stubs`/`types-tqdm`; `uv run mypy src` -> 0 ошибок |
| Производительность | 200 бумаг x 1000 строк, `max_number=3`: 0.6 c (векторные метрики) против 1.6 c (скалярный путь) |
| build | wheel 28 КБ (24 файла, `py.typed`), sdist 33 КБ (без `prices.csv`, `uv.lock`, `plan.md`) |
| CI / pre-commit | `.github/workflows/ci.yml` (jobs: quality / build / examples), `.pre-commit-config.yaml` |

Команды воспроизведения исходных дефектов (до этапа 0):

```powershell
python -c "import bfportf.targets"                                  # ModuleNotFoundError: scipy
python -c "import numpy as np; print(hasattr(np, 'in1d'))"           # False
```

Замеры производительности (после этапа 0):

| Вселенная | Время `find_portfolios(max_number=3)` |
|---|---|
| 100 бумаг x 500 строк | 0.2 c |
| 200 бумаг x 1000 строк | 0.9 c |
| ~700 бумаг x ~4000 строк (пример 01) | ~40-60 c (экстраполяция) |

Вывод: производительность приемлемая, оптимизация — задача P2, а не блокер.

---

## Реестр дефектов

### P0 — блокеры (без них библиотека не работает)

| ID | Файл:строка | Дефект | Воспроизведение | Статус |
|---|---|---|---|---|
| P0-1 | `optimization.py:83,115,120` | `np.in1d` удалён в NumPy 2.0 | `AttributeError: module 'numpy' has no attribute 'in1d'` | `[x]` |
| P0-2 | `targets.py:3`, `pyproject.toml:27-31` | `scipy` импортируется, но не объявлен в `dependencies` (в `uv.lock` тоже нет) | `ModuleNotFoundError: No module named 'scipy'` | `[x]` |
| P0-3 | `report.py:81-83` | `custom_formatting` пишет строку в float-колонку | `TypeError: Invalid value '44.03%' for dtype 'float64'` (pandas >= 2.2) | `[x]` |
| P0-4 | `pyproject.toml` | `black target-version = ["py37"]` недопустим в black >= 24, версия не запинена. **Решение:** black удалён, форматирование — `ruff format` | `uv run ruff format --check .` -> `10 files already formatted` | `[x]` |
| P0-5 | `pyproject.toml` | Конфиг ruff в устаревшем формате (верхнеуровневые `select/ignore/isort/per-file-ignores`) | перенесено в `[tool.ruff.lint]`; `uv run ruff check .` -> `All checks passed` | `[x]` |
| P0-6 | `pyproject.toml:10,16-26` | `requires-python = ">=3.7"` и классификаторы 3.7/3.8 при numpy 2 (3.7/3.8 EOL) | `>=3.9`, классификаторы 3.9-3.13, hatch-окружения удалены | `[x]` |

### P1 — корректность расчётов и методология

| ID | Файл:строка | Дефект | Статус |
|---|---|---|---|
| P1-1 | `targets.py:139-140` | Off-by-one в CAGR: `len(prices)` включает синтетическую точку `t0`, поэтому `periods` завышен. Замер: 11.153% вместо 11.274% для 1%/день x 252 | `[x]` |
| P1-2 | `targets.py:42,54,80-81,124`; `optimization.py:46-56` | Нет защиты от вырожденных данных: `profit_factor`/`sortino`/`recovery_factor` дают `inf`, `skew`/`kurtosis`/`pearsonr` на константе — `nan`. В `find_next_symbol` `np.argmax` на массиве `nan` выбирает индекс 0 (случайный кандидат) | `[x]` |
| P1-3 | `optimization.py:54` | `math.isclose` с `rel_tol=1e-9` как критерий остановки: для больших значений улучшение игнорируется, для малых — шум считается улучшением | `[x]` |
| P1-4 | `utils.py:113-116` | Неявная нормализация: `get_portfolio_return` всегда делит на `sum(weights)` (портфель принудительно 1x). Для одного символа деление полностью отменяет `scale_return_data`; веса влияют только на пропорции. Требуется явный параметр `normalize` и документирование | `[x]` |
| P1-5 | `utils.py:113-115` | Дефолтные веса строятся по всем колонкам `returns`, не по `portfolio_symbols`; нет валидации — пользователь получает «сырой» `KeyError` от pandas | `[x]` |
| P1-6 | `report.py:27,44-52` | Несогласованный контракт кастомных метрик: встроенные получают `pd.Series`, документация обещает `np.ndarray` | `[x]` |
| P1-7 | `targets.py:57-82,45-54` | Несогласованность метрик: `sharpe` — `ddof=1`, `sortino` — `ddof=0`; `recovery_factor` смешивает арифметическую сумму return-ов и геометрическую просадку; нет risk-free ставки | `[x]` |
| P1-8 | `utils.py:119-144` | `calculate_metric` дублирует `get_portfolio_return`, типизирован под `pd.Series` (ndarray ломается), используется только в примере 02 | `[x]` |
| P1-9 | `utils.py:14`; `optimization.py:17,29-30`; `report.py:38`; `targets.py:57,70` | Расхождения docstring/типов (имена параметров, порядок элементов в кортеже, `periods: int` при проверке `is None`) | `[x]` |
| P1-10 | `optimization.py:180-184` | `groupby(...).first()` в старых pandas пропускает NaN-строки; `symbols` хранится ndarray в object-колонке — неудобный контракт | `[x]` |
| P1-11 | `report.py:62-65` | Бенчмарк считается с весами портфеля (работает «случайно» — разматывает бенчмарк в сырые доходности); поведение не документировано | `[x]` |
| P1-12 | `optimization.py:186` | `reset_index(inplace=True)` даёт `Pandas4Warning` на pandas 3.1 (inplace удаляется в pandas 4). **Исправлено в этапе 0** | `[x]` |

### P2 — архитектура, инфраструктура, документация

| ID | Область | Дефект / задача | Статус |
|---|---|---|---|
| P2-1 | Тесты | `tests/` пуст (только `__init__.py`) — покрытие 0% | `[x]` |
| P2-2 | CI | `.github` отсутствует, workflow был удалён в коммите `c3240e1` | `[x]` |
| P2-3 | Метаданные | URL-ы ведут на `github.com/unknown/bfportf` (`pyproject.toml:34-36`), автор-заглушка `U.N. Owen <void@some.where>` (`__about__.py:1`), `license = "MIT"` требует `hatchling>=1.27` (PEP 639), build-backend не запинен | `[x]` |
| P2-4 | Данные примеров | `examples/prices.csv` — 14.5 МБ в git, при этом ноутбуки грузят его по HTTPS; sdist раздут (tar.gz ~7 МБ); output-ы ноутбуков не стрипнуты (NB02 — 1 МБ) | `[x]` |
| P2-5 | Тулчейн | Двойственность: `uv.lock` в репо, а задачи — в hatch-env; нет `[dependency-groups]` | `[x]` |
| P2-6 | Архитектура | `utils.py` смешивает конверсии цен, склейку портфеля и скоринг; `optimization.py` совмещает стратегию поиска и метрику; `__init__.py` пуст — нет публичного API, `py.typed`, `__version__`; нет исключений и логирования | `[x]` |
| P2-7 | Типизация | Нет `[tool.mypy]`, нет `py.typed`, `np.float64` вместо `NDArray[np.float64]` в аннотациях | `[x]` |
| P2-8 | Производительность | `np.apply_along_axis` (Python-цикл по кандидатам) + аллокация `returns + initial_returns` на каждом шаге; возможна векторизация по матрице (T x N) | `[x]` |
| P2-9 | Документация | README 26 строк без API/Quickstart/раздела о методологии весов; нет `CHANGELOG.md` | `[x]` |
| P2-10 | Линт примеров | ruff 0.17 линтит `.ipynb`: в `examples/` 16 замечаний (`E402`/`T201`/`E501`). Временно `extend-exclude = ["examples"]`; включить обратно после `nbstripout` (этап 3) | `[-]` (временно) |

---

## Этап 0 — блокеры (0.1.2)

**Цель:** пакет устанавливается и работает в чистом окружении на актуальных numpy/pandas; линтер запускается без ошибок конфигурации.

### Чек-лист

- `[x]` **P0-1.** `np.in1d` → `np.isin` в `optimization.py:83,115,120`.
- `[x]` **P0-2.** `scipy` в `[project].dependencies`; нижние границы: `numpy>=1.22`, `pandas>=1.5`, `scipy>=1.8`, `tqdm>=4.64`.
- `[x]` **P0-3.** `create_report`: `report_data.astype(object)` перед применением `custom_formatting`.
- `[x]` **P0-4.** Переход на `ruff format`: `black` удалён из зависимостей, секция `[tool.black]` удалена.
- `[x]` **P0-5.** Конфиг ruff → `[tool.ruff.lint]`, `[tool.ruff.lint.isort]`, `[tool.ruff.lint.per-file-ignores]`, `target-version = "py39"`.
- `[x]` **P0-6.** `requires-python = ">=3.9"`, классификаторы 3.9-3.13; hatch-окружения и матрица удалены (ADR-2), добавлен `[dependency-groups] dev`.
- `[x]` Смоук-тест `tests/test_smoke.py`: импорт пакета + `find_portfolios` + `create_report` (включая `custom_formatting`).
- `[x]` **ADR-2.** `uv lock` + `uv sync --group dev`; `uv.lock` актуализирован.
- `[x]` Версия `0.1.2` в `src/bfportf/__about__.py`, `plan.md` обновлён.
- `[x]` Команды разработчика (`uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`) зафиксированы в README (этап 3, P2-9).
- `[x]` Коммит этапа 0 в ветке `dev` — `b237720`.

#### Замечания по реализации

- Форматирование: `black` удалён из зависимостей, используется `ruff format` (один инструмент на линт и форматирование, ADR-2); секция `[tool.black]` удалена.
- `ruff 0.17` линтит ноутбуки: в `examples/` 16 замечаний (`E402`, `T201`, `E501`) -> добавлен `extend-exclude = ["examples"]`; вернуться к линту примеров после `nbstripout` (P2-10).
- В `ignore` добавлены `FBT001`, `FBT002`, `PLR0917`: публичный API использует позиционные bool (`maximize`, `absolute`) и длинные сигнатуры; перенос в keyword-only — этап 2.
- `results.reset_index(inplace=True, drop=True)` давал `Pandas4Warning` на pandas 3.1 -> `results = results.reset_index(drop=True)` (P1-12).
- `mypy` baseline: 6 ошибок `import-untyped` (нет стабов numpy/pandas/tqdm/scipy) — закрыть в этапе 2 (P2-7).
- Автофиксы ruff (`UP006`, `UP035`, `Q000`, `RUF010`, `I001`) перевели аннотации на PEP 585 (`list`, `tuple`, `dict`) — совместимо с `requires-python >= 3.9`.

### Критерии приёмки этапа 0

| Проверка | Ожидание |
|---|---|
| `python -c "import bfportf, bfportf.targets"` | без ошибок |
| `find_portfolios` на синтетике (50 бумаг x 250 строк) | непустой DataFrame, `target_function_value` конечные |
| `create_report(..., custom_formatting=...)` | не падает, строки в отчёте |
| `uv run pytest -q` | зелёный (смоук) |
| `uv run ruff check .` | без ошибок и без deprecation-варнингов конфига |

---

## Этап 1 — корректность расчётов (0.1.3)

**Цель:** цифры в отчёте верны на аналитических тестах, вырожденные данные не ломают поиск, семантика весов документирована.

### Чек-лист

- `[x]` **P1-1.** CAGR: `periods = len(returns) / days_per_year`; тест «константа r на n периодах → (1+r)^n - 1».
- `[x]` **P1-2.** Политика NaN/inf:
  - в `targets.py` — определённое поведение для деления на ноль (`profit_factor`, `sortino`, `recovery_factor`) и для константного ряда (`skew`, `kurtosis`, `autocorrelation`), `np.errstate`;
  - в `find_next_symbol` — маска `np.isfinite`, кандидаты с невалидным значением исключаются из выбора;
  - `validate_returns()` — проверка NaN/inf, нулевой дисперсии, дублей и немонотонного индекса.
- `[x]` **P1-3.** Параметр `tolerance` (abs/rel) вместо `math.isclose` по умолчанию; явная семантика остановки.
- `[x]` **P1-4.** `get_portfolio_return(..., normalize="sum"|"none")`; в docstring — что веса задают доли аллокации и нормализуются к 1 (неявное плечо), и что для одного символа нормализация отменяет `scale_return_data`.
- `[x]` **P1-5.** Валидация `portfolio_symbols`/весов → `UnknownSymbolError` с понятным сообщением вместо `KeyError`.
- `[x]` **P1-6.** Единый контракт кастомных метрик: `Callable[[np.ndarray], float]` (встроенные и пользовательские одинаково); поправить типы в `report.py`.
- `[x]` **P1-7.** Согласовать метрики: единый `ddof`, параметр `risk_free`, `recovery_factor` считать в одних единицах (числитель — `pnl_target`).
- `[x]` **P1-8.** `calculate_metric`: переиспользовать `get_portfolio_return`, принимать `pd.Series | np.ndarray`, либо помечать `deprecated` в пользу нового API (решение зафиксировать в журнале).
- `[x]` **P1-9.** Привести docstring/аннотации в соответствие с фактическим поведением (пп. из реестра).
- `[x]` **P1-10.** `groupby(..., dropna=False)` в `find_portfolios`; решить, как отдавать `symbols` (tuple vs ndarray).
- `[x]` **P1-11.** `create_report`: бенчмарк считается с собственными весами (или параметр `benchmark_weights`), поведение документировано.
- `[x]` Тесты этапа 1 (см. «Тестовый набор»): `test_targets.py`, `test_utils.py`, `test_report.py`, `test_optimization.py`.
- `[x]` Добавить `pytest-cov` в `[dependency-groups] dev` (нужен для `uv run pytest --cov` в критериях приёмки).
- `[x]` Поднять версию до `0.1.3`, обновить `plan.md`, коммит в `dev`.

#### Замечания по реализации

- **Решение по политике `inf`/`nan`** (был открытый вопрос): неопределённые значения метрик теперь `np.nan`; кандидаты с нефинитным значением исключаются из выбора в `find_next_symbol`; строки с `nan` в результате уходят в конец сортировки (`na_position="last"`).
- **Решение по `normalize`** (был открытый вопрос): дефолт `"sum"` сохранён ради совместимости 0.1.x (ADR-1), режим `"none"` доступен явно.
- **P1-10:** `symbols` в результате остаются `np.ndarray` (совместимость с примерами и `calculate_metric`); переход на `tuple` / `PortfolioResult` — этап 2.
- **Изменения значений метрик** (осознанно, зафиксировать в `CHANGELOG` на этапе 3): `sortino_target` использует `ddof=1` (было 0), `recovery_factor_target` считает компаундированный PnL в числителе (было арифметическая сумма). `pnl_target`, `max_drawdown_target`, `sharpe_target` значения не изменили.
- Константные и почти константные ряды определяются с относительным допуском `1e-12`: иначе `np.std(np.full(n, 0.01))` даёт около `1e-18` и Sharpe «взрывается» до `1e16`.
- `find_portfolios(validate=True)` по умолчанию проверяет входные данные; для потоковых/шумных данных можно передать `validate=False`.
- `pytest-cov` добавлен в dev-группу; покрытие снимается `uv run pytest --cov=bfportf` (конфиг `[tool.coverage.*]` не менялся, `branch = true`).

### Критерии приёмки этапа 1

| Проверка | Ожидание |
|---|---|
| `pytest -q` | зелёный, покрытие `src/bfportf` >= 80% |
| Аналитические тесты метрик | CAGR/PnL/MaxDD/Sharpe совпадают с эталоном |
| Вырожденные входы | нет `RuntimeWarning`, нет «случайного» выбора кандидата |
| `mypy src` | без новых ошибок относительно этапа 0 |

---

## Этап 2 — архитектура (0.2.0)

**Цель:** слоистая структура пакета, публичный API, расширяемость (метрики/стратегии поиска) при сохранении обратной совместимости.

### Целевая структура

```
src/bfportf/
  __init__.py        # публичный API, __all__, __version__
  py.typed
  exceptions.py      # BfPortfError -> DataValidationError / UnknownSymbolError / MetricError
  data.py            # load_returns(prices), validate_returns(), sanitize_returns()
  scaling.py         # scale_return_data, add_inverted_return, normalize_weights
  portfolio.py       # get_portfolio_return(normalize=...), aggregate_metric
  metrics/
    base.py          # Metric(name, func, maximize, vectorized) + Protocol
    basic.py         # pnl, cagr, max_drawdown, profit_factor, recovery_factor, sharpe, sortino, skew, kurtosis, autocorr
    registry.py      # METRICS = {"sharpe": Metric(..., maximize=True), "max_drawdown": Metric(..., maximize=True, # negative values)}
  search/
    base.py          # SearchStrategy Protocol, SearchConfig, PortfolioResult
    greedy.py        # текущий forward selection
    exhaustive.py    # эталон для тестов на малой вселенной
  report.py          # create_report (bc) + format_report
  utils.py           # только price <-> return, drawdown
```

### Чек-лист

- `[x]` **P2-6.** Разнести код по слоям согласно схеме выше; `utils.py` оставить только с конверсиями цен/return и `drawdown_pct`.
- `[x]` **P2-6.** `Metric` как объект: `name`, `func`, `maximize`, опциональный векторный `vectorized(matrix) -> vector`; реестр `METRICS`.
- `[x]` **P2-6.** `find_portfolios(returns, metric: str | Metric | Callable, ...)`; при `Callable` параметр `maximize` сохраняется (обратная совместимость).
- `[x]` **P2-6.** `SearchStrategy` + `SearchConfig(max_number, tolerance, normalize, min_weight)` + `PortfolioResult(symbols, weights, value, metric_name)`; DataFrame — только представление результата.
- `[x]` **P2-6.** Публичный API в `__init__.py`: `find_portfolios`, `create_report`, `METRICS`, `PortfolioResult`, `__version__`; добавить `py.typed`.
- `[x]` **P2-6.** Явные исключения вместо `KeyError`/`ValueError` из pandas/numpy.
- `[x]` **P2-6.** Логирование через `logging.getLogger("bfportf")`; `verbose` остаётся тонким шимом над прогресс-баром/логгером.
- `[x]` **P2-8.** Векторизация метрик по матрице кандидатов (T x N), `np.add(..., out=...)` вместо аллокаций, кэш `cumprod` для компаундирующих метрик.
- `[-]` Deprecation-шимы — **отменено (ADR-1)**: breaking changes допускаются; вместо шимов — раздел `Breaking changes` в `CHANGELOG.md` и обновление примеров/README.
- `[x]` **ADR-1.** Обновить `examples/*.ipynb` и `README.md` под новый API 0.2.0.
- `[x]` **ADR-3.** `search/`: `GreedySearch` (дефолт), `BeamSearch(width=5)`, `ExhaustiveSearch(max_combinations=2_000_000)`; оценка числа комбинаций до запуска.
- `[x]` **ADR-3.** `exhaustive` как эталон в тестах: на 6 бумагах (`max_number` 1-3) результат равен истинному максимуму, greedy/beam сравниваются с ним.
- `[x]` **ADR-4.** `find_portfolios` возвращает `list[PortfolioResult]`; `results_to_frame()` для табличного вида; `maximize=None` -> направление из `Metric`.
- `[x]` **P2-8.** Векторный fast-path: `Metric.vectorized` для компаундирующих и моментных метрик, `np.add(..., out=...)`, кэш `cumprod`; замер ускорения.
- `[x]` Тесты на новые слои (`test_metrics_registry.py`, `test_search_strategies.py`, `test_api.py`).
- `[x]` Поднять версию до `0.2.0`, добавить `CHANGELOG.md`, обновить `plan.md`, коммит в `dev`.

#### Замечания по реализации

- **Направление `max_drawdown` — `maximize=True`** (уточнение ADR-3): метрика возвращает **отрицательный** процент, поэтому лучший портфель имеет наибольшее значение. При `maximize=False` поиск выбирал бы самую глубокую просадку. Отражено в `CHANGELOG` (0.2.0).
- Векторный путь должен учитывать стартовый капитал: `_growth` теперь добавляет ведущую строку единиц, иначе первая доходность не попадает в просадку и `max_drawdown`/`recovery_factor` расходятся со скалярной версией (нашёл тест «vectorized == scalar»).
- `np.add(..., out=...)` не применялся: основной выигрыш дала векторизация (2.7x), экономия от буферов в пределах шума; при необходимости — отдельная задача.
- mypy: `python_version = "3.10"`, так как mypy 2.x больше не поддерживает 3.9 как target; сам пакет по-прежнему `requires-python >= 3.9`.
- `scipy-stubs` не добавлен (требует Python >= 3.10.1) — для `scipy.*` настроен `ignore_missing_imports`; `numpy`/`pandas`/`tqdm` покрыты стабами.
- `bfportf.targets` и `bfportf.optimization` оставлены тонкими фасадами (re-export и вызов стратегии), чтобы импорты из примеров и внешнего кода продолжали работать, хотя ADR-1 этого не требовал.
- Ноутбуки перегенерированы программно и очищены от output-ов (заодно закрыта часть P2-4); данные по-прежнему загружаются по HTTPS.

### Критерии приёмки этапа 2

| Проверка | Ожидание |
|---|---|
| Совместимость (ADR-1) | breaking changes допустимы: важны корректные значения метрик, а не сохранение старых сигнатур |
| `from bfportf import find_portfolios, create_report, METRICS, __version__` | работает |
| Ноутбуки-примеры и README | обновлены под новый API и выполняются |
| Производительность 200x1000, `max_number=3` | не медленнее 0.9 c (цель — ускорение >= 2x) |
| `uv run pytest -q`, покрытие | зелёный, >= 80% |
| `find_portfolios` | возвращает `list[PortfolioResult]`, лучший первым |
| `BeamSearch(width=1)` | совпадает с `GreedySearch` |
| `ExhaustiveSearch` на 6 бумагах | равен истинному максимуму (brute force) |
| `uv run mypy src` | 0 ошибок |
| Производительность 200 x 1000 | 0.6 c против 1.6 c скалярного пути (2.7x) |

---

## Этап 3 — упаковка и документация (0.2.x)

**Цель:** воспроизводимая сборка, зелёный CI, чистый репозиторий, документация для пользователя.

### Чек-лист

- `[x]` **P2-2 / ADR-2.** `.github/workflows/ci.yml`: matrix Python 3.10-3.13, `uv sync --locked`, `uv run pytest --cov`, `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src`; отдельный job — `uv build` и установка wheel в чистом venv.
- `[x]` **P2-2.** `pre-commit` (запуск через `uv run pre-commit`): `astral-sh/uv-pre-commit` (`uv-lock`), `ruff`, `ruff-format`, `nbstripout`, проверка whitespace/EOF.
- `[x]` **P2-3.** Метаданные: реальные URL-ы и автор, `license` по PEP 639 + `hatchling>=1.27` в `build-system.requires`, актуальные классификаторы и `keywords`.
- `[x]` **P2-5.** `[dependency-groups] dev` (pytest, coverage, ruff, mypy), `[project.optional-dependencies] examples` (scikit-learn, quantstats, jupyter); `uv lock`; `uv.lock` в git; команды разработчика — через `uv run`.
- `[x]` **P2-4.** Убрать `examples/prices.csv` из индекса git (`.gitignore`), исключить из sdist (`[tool.hatch.build.targets.sdist] exclude`); ноутбуки грузят данные по URL.
- `[x]` **P2-4.** Стрипнуть output-ы ноутбуков (`nbstripout`), NB02 уменьшится с ~1 МБ.
- `[x]` **P2-7.** `[tool.mypy]` (`disallow_untyped_defs`, `warn_unused_ignores`), `py.typed`, переход на `NDArray[np.float64]` в аннотациях.
- `[x]` **P2-9.** README: Quickstart (код из примера 01), раздел «Методология: веса, нормализация, плечо», раздел про выбор метрики и `maximize`.
- `[x]` **P2-9.** `CHANGELOG.md` (Keep a Changelog), версии по этапам 0.1.2 / 0.1.3 / 0.2.0.
- `[x]` Проверка сборки (ADR-2): `uv build`, установка wheel в чистом venv (`uv venv` + `uv pip install`), прогон смоук-теста.
- `[x]` Обновить `plan.md` (финальный статус), коммит в `dev`.

#### Замечания по реализации

- **ADR-5:** релиза в PyPI нет. Артефакты `dist/*.whl` и `dist/*.tar.gz` прикладываются к GitHub-релизу; README больше не содержит PyPI-бейджей, установка описана как `pip install <wheel>`.
- sdist ужат с 555 КБ до 33 КБ: исключены `uv.lock` (1.7 МБ), `plan.md`, `examples/prices.csv`, `.github`, `.pre-commit-config.yaml`.
- Wheel (28 КБ, 24 файла) содержит `py.typed` и все модули пакета; установка в чистое venv подтягивает только `numpy`/`pandas`/`scipy`/`tqdm`.
- `examples` extra тянет `jupyter`, `quantstats`, `scikit-learn` — в `uv.lock` 251 пакет с учётом extra; CI-job `examples` проверяет только резолв, чтобы не устанавливать Jupyter на каждый прогон.
- `examples/prices.csv` удалён и из индекса, и с диска; исходные данные остаются в истории git (`314397c`) и используются только как последний резерв конвертации. Рабочий формат — `assets/prices.parquet` (4.5 МБ, `float32`), он не в git (`/assets/*.parquet`), исключён из sdist и публикуется ассетом GitHub-релиза; чтение parquet требует `pyarrow` (в `dependencies`).
- Версии `rev` в `.pre-commit-config.yaml` зафиксированы вручную; обновление — `uv run pre-commit autoupdate`.
- ruff по-прежнему исключает `examples/` (P2-10): в ячейках примеров осмысленный порядок импортов и длинные URL.

### Критерии приёмки этапа 3

| Проверка | Ожидание |
|---|---|
| CI на GitHub | зелёный на 3.10-3.13 (через `uv sync --locked`) |
| `uv build` | успешно; wheel < 100 КБ, sdist без `prices.csv` |
| Размеры артефактов | wheel 28 КБ (24 файла, `py.typed`), sdist 33 КБ (без `prices.csv`, `uv.lock`, `plan.md`) |
| `uv lock --check` | lock актуален, включает `examples` extra |
| Установка wheel в чистом venv | `import bfportf` + смоук-тест проходят |
| Репозиторий | нет данных > 1 МБ, ноутбуки без output-ов |
| README | позволяет пройти путь «установка -> поиск портфеля -> отчёт» без чтения кода |

---

## Тестовый набор (целевое состояние)

```
tests/
  conftest.py            # synthetic_returns(n_symbols, n_rows, vol_range, seed), фикстуры small_universe
  test_smoke.py          # этап 0: import пакета, find_portfolios, create_report
  test_utils.py          # этап 1
  test_targets.py        # этап 1
  test_optimization.py   # этап 1
  test_report.py         # этап 1
  test_api.py            # этап 2: публичный API, реестр METRICS, обратная совместимость импортов
```

| Файл | Кейсы | Этап |
|---|---|---|
| `test_utils.py` | roundtrip `price_to_return(return_to_price(r)) ≈ r`; `drawdown_pct` на известной траектории; `scale_return_data` — сумма весов = 1, веса обратно пропорциональны `mean|r|`; `get_portfolio_return` — равные веса / inverse-vol / `normalize="none"`; ошибка на неизвестном символе | 1 |
| `test_targets.py` | PnL `[0.1, 0.1] -> 0.21`; MaxDD на монотонном и V-образном ряде; **CAGR константы = (1+r)^n - 1** (регрессия P1-1); `profit_factor` без убытков; `sortino` без отрицательных; `recovery_factor` без просадки; `autocorrelation` на AR(1) ~ phi; согласованность ddof у Sharpe/Sortino | 1 |
| `test_optimization.py` | exhaustive-эталон на вселенной из 6 бумаг для `max_number = 1, 2, 3`; `max_number=1`; вселенная из 1 бумаги; пустой список кандидатов; колонка из NaN; поведение `tolerance`; детерминизм повторного запуска | 1 |
| `test_report.py` | значения отчёта = прямой расчёт метрик; колонка `Diff`; `benchmark_symbols` vs `custom_benchmark`; `custom_formatting` возвращает строки (регрессия P0-3); неизвестная метрика в `custom_formatting` игнорируется | 1 |
| `test_api.py` | `bfportf.__version__`; экспорты из `__init__`; реестр `METRICS` содержит корректный флаг `maximize` для каждой метрики; старые импорты продолжают работать | 2 |

---

## Журнал изменений

| Дата | Этап | Изменение | Проверки | Коммит |
|---|---|---|---|---|
| 2026-11-04 | Аудит | Проведён анализ кода и окружения, дефекты воспроизведены, создан `plan.md` | `python -c "import bfportf.targets"` -> ModuleNotFoundError (scipy); `hasattr(np, "in1d")` -> False; `create_report(custom_formatting=...)` -> TypeError | — |
| 2026-11-04 | Решения | Зафиксированы ADR-1 (breaking changes в 0.2.0) и ADR-2 (переход на `uv`, hatch-окружения удаляются) | — | — |
| 2026-11-04 | 0 | Блокеры закрыты: `np.isin` вместо `np.in1d`; `scipy>=1.8` + нижние границы зависимостей; `custom_formatting` через `astype(object)`; ruff-конфиг в `[tool.ruff.lint]`; переход на `ruff format` (black удалён); `requires-python>=3.9`; удалены hatch-окружения, добавлен `[dependency-groups] dev`; версия 0.1.2; смоук-тесты | `uv run pytest -q` -> 4 passed; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 10 files already formatted; сквозной сценарий `scale_return_data` -> `find_portfolios` -> `create_report(custom_formatting=...)` | b237720 |
| 2026-11-04 | 1 | Корректность: фикс CAGR (off-by-one); политика NaN/inf в метриках; isfinite-фильтр кандидатов; параметр `tolerance`; явный `normalize` в `get_portfolio_return`; `exceptions.py` (`DataValidationError`/`UnknownSymbolError`); `validate_returns()`; согласованы `ddof`/`risk_free`; recovery factor через компаундированный PnL; `benchmark_weights`; полный набор тестов (85) | `uv run pytest -q --cov=bfportf` -> 85 passed, 95%; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 17 files already formatted; `uv run mypy src` -> 6 `import-untyped` (baseline) | f92620d |
| 2026-11-04 | 2 | Архитектура 0.2.0: слои `data`/`scaling`/`portfolio`/`metrics`/`search`; реестр `METRICS`; стратегии greedy/beam/exhaustive; `list[PortfolioResult]`; публичный API + `py.typed`; векторные метрики (2.7x); mypy чистый; примеры и README обновлены; добавлен `CHANGELOG` | `uv run pytest -q --cov=bfportf` -> 124 passed, 96%; `uv run ruff check .` -> All checks passed; `uv run ruff format --check .` -> 34 files already formatted; `uv run mypy src` -> Success | 780f764 |
| 2026-11-04 | 3 | Упаковка: CI-workflow (3 job, matrix 3.10-3.13), pre-commit, метаданные PEP 639, `examples` extra, `prices.csv` убран из git, sdist/wheel ужаты, README/CHANGELOG обновлены, ADR-5 (без PyPI) | `uv build` -> wheel 28 КБ + sdist 33 КБ; установка wheel в чистом venv -> `0.2.0 35 sharpe`; `uv lock --check` -> OK; `uv run pytest -q` -> 124 passed; ruff/format/mypy чисто | c98752f |
| 2026-11-05 | 3 | Примеры: ноутбуки 01/02 переведены на API 0.2, добавлен `03 Metrics and strategies`; данные — локальная копия или pinned-коммит `314397c`; `scikit-learn` убран из `examples` extra; README/CHANGELOG обновлены | nbclient-прогон: 01 — 7.8 c, 02 — 12.6 c, 03 — 3.7 c, ошибок нет; `uv lock --check` -> OK; `uv run pytest -q` -> 124 passed; ruff/format/mypy чисто | d77b4fb |
| 2026-11-05 | 3 | Данные примеров переведены на parquet: `assets/prices.parquet` (4.5 МБ, `float32`, 500 символов) вместо `examples/prices.csv`; файл не в git, исключён из sdist, публикуется ассетом GitHub-релиза; ноутбуки ищут локальную копию (`assets/`, рядом, `../assets`), затем качают `releases/latest/download/prices.parquet`, затем конвертируют CSV из `314397c`; `pyarrow>=21.0.0` в зависимостях | nbclient: cwd=`examples` (8.0 / 13.6 / 4.0 c), cwd=корень и cwd=пустой каталог (fallback CSV -> parquet, 5.8 c) — везде 2114x500 и Sharpe 2.352004; `uv lock --check` -> exit 0; pytest 124 passed; ruff/format/mypy чисто; wheel 28.6 КБ с `Requires-Dist: pyarrow`, sdist 38 КБ без parquet/csv | 4b85aa7 |

---

## Правила ведения документа

1. Перед началом работы над пунктом: `[ ]` -> `[~]`.
2. После проверки критериев: `[~]` -> `[x]`, обновление таблицы «Статус» и строки в «Журнале изменений» (дата, этап, что изменено, чем проверено, хеш коммита).
3. Если пункт отменён или перенесён — `[-]`/перенос в другой этап + причина в журнале.
4. Новые находки, обнаруженные по ходу работ, добавляются в «Реестр дефектов» с новым ID (`P1-12`, `P2-10`, ...) и распределяются по этапам.
5. `plan.md` коммитится вместе с изменениями кода того же этапа.

---

## Открытые вопросы

**Решено:**

- ~~Обратная совместимость~~ -> **ADR-1**: breaking changes допускаются в 0.2.0, шимы не делаем.
- ~~Тулчейн~~ -> **ADR-2**: полностью переходим на `uv`.

## Открытые вопросы

**Решено:**

- ~~Обратная совместимость~~ -> **ADR-1**: breaking changes допускаются в 0.2.0, шимы не делаем.
- ~~Тулчейн~~ -> **ADR-2**: полностью переходим на `uv`.
- ~~Стратегии поиска~~ -> **ADR-3**: `greedy` / `beam` / `exhaustive` — публичные классы стратегий, `greedy` по умолчанию.
- ~~Формат результата поиска~~ -> **ADR-4**: `find_portfolios` возвращает `list[PortfolioResult]`, DataFrame — через `results_to_frame`.
- ~~Политика `inf`/`nan`~~ -> реализовано в 0.1.3: метрики возвращают `np.nan`, кандидаты с нефинитным значением отбрасываются.
- ~~Семантика весов~~ -> реализовано в 0.1.3: дефолт `normalize="sum"`, режим `"none"` доступен явно.
- ~~Публикация в PyPI~~ -> **ADR-5**: релиза в PyPI не будет, распространяем `wheel`/`sdist` через GitHub-релизы.

Открытых вопросов не осталось.