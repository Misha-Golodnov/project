# Text Paraphraser

Веб-приложение для перефразирования текста на русском языке с использованием модели `rut5-base-paraphraser`.

## Возможности

- Перефразирование текста на русском языке
- Настраиваемые параметры генерации (количество вариантов, beam search, temperature)
- Веб-интерфейс и REST API
- Поддержка GPU (CUDA)

## Установка

### Docker

```bash
docker build -t text-paraphraser .
docker run -p 8000:8000 --gpus all text-paraphraser
```

### Локально

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Использование

- Веб-интерфейс: http://localhost:8000/static/index.html
- API документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API

### POST /paraphrase

```json
{
  "text": "Ваш текст для перефразирования",
  "num_return_sequences": 1,
  "num_beams": 5,
  "temperature": 1.0
}
```
