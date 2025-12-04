# Используем легкий образ
FROM python:3.11-slim

# Отключаем буферизацию вывода (чтобы логи видели сразу)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Устанавливаем системные зависимости
# libpq-dev нужен для сборки psycopg2 (если он используется где-то неявно) или других pg либ
# build-essential (gcc) нужен для сборки некоторых python пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Запуск
CMD ["python", "bot.py"]
