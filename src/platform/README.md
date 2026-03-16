# Platform Navigation

`src/platform/` — это инфраструктурная зона baseline. Она отвечает за запуск пайплайна, артефакты, контракты и локальную валидацию.

Обычно вам не нужно менять эти файлы, если задача состоит в улучшении качества решения. Для общей карты репозитория см. [`docs/baseline/ONBOARDING.md`](../../docs/baseline/ONBOARDING.md), а для редактируемой solution-зоны см. [`src/competition/README.md`](../competition/README.md).

## Как читать эту директорию

- `cli/` — команды запуска и загрузка конфига.
- `pipeline/` — orchestration, stage chain и runtime workflows.
- `core/` — dataset loader, submission contract, artifacts, metrics.
- `infra/` — низкоуровневые технические утилиты.

Если хотите быстро понять исполнение пайплайна, обычно достаточно такого порядка:

1. `cli/entrypoint.py`
2. `pipeline/models.py`
3. `pipeline/orchestrator.py`
4. `pipeline/stages/`
5. `core/dataset.py`

## Ключевые файлы

- `cli/entrypoint.py` — основная CLI-точка входа для `run` и `validate`.
- `cli/config_loader.py` — загрузка YAML-конфигов с `imports`.
- `pipeline/models.py` — список стадий и их зависимости.
- `pipeline/orchestrator.py` — runner, cache-aware execution и запись run metadata.
- `pipeline/runtime.py` — сборка runtime dataset из сырых файлов и кэша.
- `pipeline/workflows/local_validation.py` — локальная псевдо-валидация качества.
- `core/dataset.py` — загрузка и нормализация входных CSV.
- `core/submission_contract.py` — строгая проверка итогового сабмита.
- `core/artifacts.py` — атомарная запись артефактов и статус стадий.

## Где `platform` вызывает `competition`

Эти файлы служат glue-code между инфраструктурой и solution-логикой:

- `pipeline/stages/build_features.py`
- `pipeline/stages/generate_candidates.py`
- `pipeline/stages/rank_and_select.py`
- `pipeline/stages/make_submission.py`
- `pipeline/workflows/local_validation.py`

Если нужно понять, как именно solution подключается к platform, смотрите сначала их.

## Что обычно не редактируют без необходимости

- `cli/`
- `core/`
- `infra/`
- `pipeline/orchestrator.py`

Менять `platform` имеет смысл только если вы осознанно хотите тронуть lifecycle пайплайна, контракты, кэширование или локальную валидацию.

## Если нужно дебажить пайплайн

- смотрите `artifacts/_meta/run.json`
- смотрите `artifacts/_meta/step_status.json`
- смотрите `logs/`
- для общей smoke-карты полезен `tests/platform/test_pipeline_smoke.py`

## Что почитать как карту архитектуры

- `tests/platform/test_pipeline_smoke.py`
- `tests/platform/test_architecture_enforcement.py`
- `tests/platform/test_submission_contract.py`
