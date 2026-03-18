# ONBOARDING

Этот baseline разделён на две зоны:

- `src/competition/` — зона решения, которую вам имеет смысл менять.
- `src/platform/` — инфраструктура пайплайна: запуск, артефакты, контракты и локальная валидация.

Если нужна короткая эвристика:

- хотите улучшать качество рекомендаций — начинайте с [`src/competition/README.md`](../../src/competition/README.md)
- хотите понять, как это исполняется в рантайме — читайте [`src/platform/README.md`](../../src/platform/README.md)

## Как устроен baseline

Пайплайн идёт по пяти стадиям:

1. `prepare_data`
2. `build_features`
3. `generate_candidates`
4. `rank_and_select`
5. `make_submission`

На практике это означает такой поток:

```text
raw CSV -> data_cache.csv -> features.csv -> candidates.csv -> predictions.csv -> submission.csv
```

`platform` управляет этим жизненным циклом, а `competition` подставляет solution-логику в ключевые точки: построение признаков, генерация кандидатов и ранжирование.

## С чего начать чтение

- Запуск и артефакты: [`SETUP.md`](./SETUP.md)
- Постановка задачи и формат сабмита: [`../task/task_description.md`](../task/task_description.md)
- Схема данных: [`../task/data_description.md`](../task/data_description.md)
- Навигация по editable-зоне: [`../../src/competition/README.md`](../../src/competition/README.md)
- Навигация по infrastructure-зоне: [`../../src/platform/README.md`](../../src/platform/README.md)

## Что стоит изучить и улучшать

- `src/competition/features.py`
- `src/competition/generators/`
- `src/competition/generators/registry.py`
- `src/competition/ranking.py`

## Что не рекомендуется изменять

- `src/platform/cli/`
- `src/platform/pipeline/`
- `src/platform/core/`
- `src/platform/infra/`