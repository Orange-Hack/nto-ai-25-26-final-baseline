# Competition Navigation

`src/competition/` — это зона решения. Если вы хотите улучшать качество baseline, в большинстве случаев начинать нужно именно отсюда.

Общий контекст по всему репозиторию: [`docs/baseline/ONBOARDING.md`](../../docs/baseline/ONBOARDING.md). Если нужно понять lifecycle пайплайна и glue-code вокруг этой зоны, см. [`src/platform/README.md`](../platform/README.md).

## Куда чаще всего вносят изменения

- `features.py` — строит feature-table для следующих стадий пайплайна.
- `ranking.py` — ранжирует кандидатов, фильтрует `seen_positive` и добивает недостающие позиции fallback-логикой.
- `generators/` — здесь лежат candidate generators baseline.
- `generators/registry.py` — связывает имя генератора в конфиге с конкретной реализацией.

## Карта директории

- `features.py` — признаки для генерации кандидатов.
- `ranking.py` — финальный отбор top-k.
- `validation.py` — thin wrapper над контрактом сабмита.
- `generators/base.py` — контракт генератора.
- `generators/runner.py` — запуск генераторов и проверка их output schema.
- `generators/global_popularity.py` — глобально-популярный baseline.
- `generators/user_genre.py` — baseline по жанровому профилю пользователя.
- `generators/user_author.py` — baseline по авторскому профилю пользователя.

## Типовые сценарии

### Добавить новый генератор

1. Добавьте реализацию в `generators/`.
2. Соблюдите контракт из `generators/base.py`.
3. Зарегистрируйте генератор в `generators/registry.py`.
4. Добавьте его в YAML-конфиг эксперимента.

### Добавить новые признаки

1. Расширьте `features.py`.
2. Убедитесь, что новые признаки реально используются в генераторах или ranker'е.

### Поменять ранжирование

1. Начинайте с `ranking.py`.
2. Сохраняйте выходной контракт: на выходе должен получаться корректный top-k для каждого пользователя.

## Что обычно не нужно трогать

- `__init__.py`
- `generators/__init__.py`
- `generators/runner.py`, если вы не меняете сам контракт генераторов
- `validation.py`, если вы не меняете формат финального сабмита

## Полезно знать заранее

- Имена источников кандидатов согласованы с конфигом `ranking.source_weights` (например, `user_genre`), поэтому веса применяются прозрачно.
- Если добавляете новый генератор, держите единый идентификатор в трёх местах: `generator.name`, ключ в `GENERATOR_REGISTRY` и ключ веса в конфиге.

## На что смотреть как на примеры

- `tests/competition/test_solution_api.py`
- `tests/competition/test_solution_generators.py`
