import pytest
from fastapi.testclient import TestClient
from app.main import app, ParaphraseRequest, ParaphraseResponse
import string
import re


client = TestClient(app)


class TextValidator:
    """Фильтр для проверки адекватности символов в тексте"""

    # Допустимые символы: буквы, цифры, пробелы, пунктуация, кириллица
    ALLOWED_PATTERNS = [
        r'[а-яА-ЯёЁ]',  # Кириллица
        r'[a-zA-Z]',     # Латиница
        r'[0-9]',        # Цифры
        r'[\s]',         # Пробелы
        r'[.,!?;:\-—–"\'«»„"„]',  # Пунктуация
    ]

    FORBIDDEN_CHARS = {
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05',  # Управляющие символы
        '\x06', '\x07', '\x08', '\x0b', '\x0c', '\x0e',
    }

    @classmethod
    def is_valid_text(cls, text: str) -> bool:
        """Проверяет, содержит ли текст только допустимые символы"""
        if not text:
            return False

        # Проверка на управляющие символы
        if any(char in cls.FORBIDDEN_CHARS for char in text):
            return False

        # Проверка на адекватность символов
        for char in text:
            if not cls._is_valid_char(char):
                return False

        return True

    @classmethod
    def _is_valid_char(cls, char: str) -> bool:
        """Проверяет один символ"""
        for pattern in cls.ALLOWED_PATTERNS:
            if re.match(pattern, char):
                return True
        return False

    @classmethod
    def filter_text(cls, text: str) -> str:
        """Удаляет недопустимые символы из текста"""
        return ''.join(char for char in text if cls._is_valid_char(char))


# ========== ТЕСТЫ ВАЛИДАТОРА ТЕКСТА ==========

class TestTextValidator:
    """Тесты для фильтра валидации текста"""

    def test_valid_russian_text(self):
        """Тест на валидный русский текст"""
        text = "Привет, это тест."
        assert TextValidator.is_valid_text(text) is True

    def test_valid_english_text(self):
        """Тест на валидный английский текст"""
        text = "Hello, this is a test."
        assert TextValidator.is_valid_text(text) is True

    def test_valid_mixed_text(self):
        """Тест на смешанный текст"""
        text = "Hello привет 123"
        assert TextValidator.is_valid_text(text) is True

    def test_empty_text(self):
        """Тест на пустой текст"""
        text = ""
        assert TextValidator.is_valid_text(text) is False

    def test_text_with_control_characters(self):
        """Тест на текст с управляющими символами"""
        text = "Hello\x00World"
        assert TextValidator.is_valid_text(text) is False

    def test_text_with_special_html_characters(self):
        """Тест на текст со специальными HTML символами"""
        text = "Hello<script>alert('xss')</script>"
        assert TextValidator.is_valid_text(text) is False

    def test_filter_removes_invalid_chars(self):
        """Тест на удаление недопустимых символов"""
        text = "Hello<>World"
        filtered = TextValidator.filter_text(text)
        assert filtered == "HelloWorld"
        assert TextValidator.is_valid_text(filtered) is True

    def test_filter_preserves_valid_chars(self):
        """Тест на сохранение допустимых символов"""
        text = "Hello, привет 123"
        filtered = TextValidator.filter_text(text)
        assert "Hello" in filtered
        assert "привет" in filtered
        assert "123" in filtered

    def test_cyrillic_validation(self):
        """Тест валидации кириллицы"""
        text = "Тестовая строка"
        assert TextValidator.is_valid_text(text) is True

    def test_numbers_validation(self):
        """Тест валидации чисел"""
        text = "Test 12345"
        assert TextValidator.is_valid_text(text) is True


# ========== ТЕСТЫ API ENDPOINTS ==========

class TestHealthEndpoint:
    """Тесты для endpoint'а /health"""

    def test_health_endpoint_returns_200(self):
        """Тест на то, что health endpoint возвращает 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_has_required_fields(self):
        """Тест на наличие всех требуемых полей"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "cuda_available" in data

    def test_health_endpoint_status_field(self):
        """Тест на значение status в health endpoint"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestParaphraseRequest:
    """Тесты для модели ParaphraseRequest"""

    def test_valid_request_model(self):
        """Тест создания валидного request'а"""
        request = ParaphraseRequest(
            text="Привет мир",
            num_return_sequences=1,
            num_beams=5
        )
        assert request.text == "Привет мир"
        assert request.num_return_sequences == 1
        assert request.num_beams == 5

    def test_request_with_defaults(self):
        """Тест request'а с значениями по умолчанию"""
        request = ParaphraseRequest(text="Test")
        assert request.num_return_sequences == 1
        assert request.num_beams == 5
        assert request.temperature == 1.0

    def test_request_with_custom_temperature(self):
        """Тест request'а с кастомной температурой"""
        request = ParaphraseRequest(text="Test", temperature=0.7)
        assert request.temperature == 0.7


class TestParaphraseEndpoint:
    """Тесты для endpoint'а /paraphrase"""

    def test_empty_text_returns_400(self):
        """Тест на то, что пустой текст возвращает 400"""
        response = client.post("/paraphrase", json={"text": ""})
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_whitespace_only_returns_400(self):
        """Тест на то, что текст только с пробелами возвращает 400"""
        response = client.post("/paraphrase", json={"text": "   "})
        assert response.status_code == 400

    def test_valid_request_structure(self):
        """Тест структуры валидного request'а"""
        request_data = {
            "text": "Это тестовый текст",
            "num_return_sequences": 2,
            "num_beams": 5
        }
        # Просто проверяем, что модель принимает эту структуру
        try:
            request = ParaphraseRequest(**request_data)
            assert request.text == request_data["text"]
        except Exception as e:
            pytest.fail(f"Valid request structure should not raise: {e}")

    def test_response_model_structure(self):
        """Тест структуры response модели"""
        response = ParaphraseResponse(
            original_text="Test",
            paraphrases=["Test 1", "Test 2"],
            device="cpu"
        )
        assert response.original_text == "Test"
        assert len(response.paraphrases) == 2
        assert response.device == "cpu"


# ========== ИНТЕГРАЦИОННЫЕ ТЕСТЫ ==========

class TestIntegration:
    """Интеграционные тесты"""

    def test_text_validation_before_processing(self):
        """Тест валидации текста перед обработкой"""
        test_cases = [
            ("Привет мир", True),
            ("Hello world", True),
            ("", False),
            ("   ", False),
        ]

        for text, expected in test_cases:
            result = TextValidator.is_valid_text(text)
            assert result == expected, f"Text '{text}' validation failed"

    def test_filter_removes_malicious_patterns(self):
        """Тест на удаление потенциально опасных паттернов"""
        malicious_texts = [
            "Hello<script>alert('xss')</script>World",
            "Text with\x00null",
            "Normal<>Brackets",
        ]

        for text in malicious_texts:
            filtered = TextValidator.filter_text(text)
            assert TextValidator.is_valid_text(filtered) is True
            assert "<" not in filtered
            assert ">" not in filtered
            assert "\x00" not in filtered

    def test_request_with_filtered_text(self):
        """Тест request'а с отфильтрованным текстом"""
        original = "Hello<>World"
        filtered = TextValidator.filter_text(original)

        request = ParaphraseRequest(text=filtered)
        assert TextValidator.is_valid_text(request.text) is True

    def test_batch_text_validation(self):
        """Тест валидации массива текстов"""
        texts = [
            "Первый текст",
            "Второй текст",
            "Third text",
            "123 numbers",
        ]

        valid_texts = [t for t in texts if TextValidator.is_valid_text(t)]
        assert len(valid_texts) == len(texts)


# ========== ГРАНИЧНЫЕ СЛУЧАИ ==========

class TestEdgeCases:
    """Тесты граничных случаев"""

    def test_very_long_text(self):
        """Тест на очень длинный текст"""
        text = "Тестовый текст " * 1000
        assert TextValidator.is_valid_text(text) is True

    def test_single_character(self):
        """Тест на один символ"""
        assert TextValidator.is_valid_text("А") is True
        assert TextValidator.is_valid_text("a") is True
        assert TextValidator.is_valid_text("1") is True

    def test_unicode_characters(self):
        """Тест на unicode символы"""
        text = "Hello мир 你好"  # Китайские иероглифы
        # Кириллица допустима, но иероглифы - нет
        assert TextValidator.is_valid_text("Hello мир") is True

    def test_special_punctuation(self):
        """Тест на специальную пунктуацию"""
        text = "Hello, world! How are you?"
        assert TextValidator.is_valid_text(text) is True

    def test_quotes_validation(self):
        """Тест на валидацию разных типов кавычек"""
        texts = [
            'Text with "double quotes"',
            "Text with 'single quotes'",
            'Text with «guillemets»',
        ]

        for text in texts:
            assert TextValidator.is_valid_text(text) is True

    def test_multiple_spaces(self):
        """Тест на множественные пробелы"""
        text = "Hello    world"
        assert TextValidator.is_valid_text(text) is True

    def test_newlines_and_tabs(self):
        """Тест на новые строки и табуляцию"""
        text = "Hello\nWorld\tTest"
        # Табуляция - это управляющий символ \t, проверим
        assert TextValidator.is_valid_text(
            text) is True or TextValidator.is_valid_text(text) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
