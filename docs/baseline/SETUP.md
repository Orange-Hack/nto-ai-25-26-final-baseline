# SETUP

## Prerequisites

1. Установка uv (https://docs.astral.sh/uv/getting-started/installation/)
2. Запускайте команды из корня репозитория.
3. В `data/` лежат входные CSV с ODS:

```text
data/
  authors.csv
  book_genres.csv
  editions.csv
  genres.csv
  interactions.csv
  targets.csv
  users.csv
```

## Быстрый запуск

Полный прогон:

```bash
uv run python -m src.platform.cli.entrypoint run --config configs/experiments/baseline.yaml
```

Перезапуск с конкретной стадии:

```bash
uv run python -m src.platform.cli.entrypoint run --config configs/experiments/baseline.yaml --stage generate_candidates
```

Локальная валидация (псевдо-инцидент):

```bash
uv run python -m src.platform.cli.entrypoint validate --config configs/experiments/baseline.yaml
```

## Где смотреть результаты

Итоговый сабмит: `artifacts/submission.csv`

Промежуточные артефакты: `artifacts/*.csv`

Метаданные запуска:
  - `artifacts/_meta/run.json`
  - `artifacts/_meta/step_status.json`

Логи: `logs/`


После первого успешного запуска переходите к [`ONBOARDING.md`](ONBOARDING.md), чтобы понять, как эффективно улучшать baseline.
