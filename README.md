# Lang Teacher Agent

Базовый каркас Python-приложения с поддержкой Docker и PostgreSQL.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python src/main.py
```

## Тесты

```bash
pytest
```

## Docker

Собрать и запустить сервисы (приложение + PostgreSQL):

```bash
docker compose up --build
```

## GitHub Actions

При пуше в репозиторий запускаются две задачи: проверка тестов и пробная сборка Docker-образа. Файл конфигурации находится в `.github/workflows/ci.yml`.

