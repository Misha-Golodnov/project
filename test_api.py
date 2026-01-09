"""
Тесты к API с моделью.
Требует запущенного сервера на http://localhost:8000
"""
import os
import requests
import pytest
import time

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _is_api_available(timeout: float = 1.5) -> bool:
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _wait_for_api(total_wait_seconds: float = 5.0) -> bool:
    deadline = time.time() + total_wait_seconds
    while time.time() < deadline:
        if _is_api_available():
            return True
        time.sleep(0.25)
    return _is_api_available(timeout=2.5)


@pytest.fixture(autouse=True, scope="module")
def _require_real_api():
    """Skip these integration tests unless the real API is reachable."""
    run_real = os.getenv("RUN_REAL_API_TESTS") in {"1", "true", "True", "yes", "YES"}
    if run_real:
        return

    wait_seconds = float(os.getenv("API_WAIT_SECONDS", "5"))
    if _wait_for_api(total_wait_seconds=wait_seconds):
        return

    pytest.skip(
        f"Real API tests require a running server at {BASE_URL}. "
        "Start it with `uvicorn app.main:app --host 0.0.0.0 --port 8000` "
        "or set RUN_REAL_API_TESTS=1 to force running these tests."
    )


class TestRealAPIRequests:
    """Тесты к работающему API"""

    @staticmethod
    def test_health_check():
        """Проверка здоровья сервера"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        print(f"Health check passed: {data}")

    @staticmethod
    def test_paraphrase_simple_text():
        """Тест парафраза простого текста"""
        payload = {
            "text": "Привет мир",
            "num_return_sequences": 1,
            "num_beams": 5,
            "temperature": 1.0
        }
        response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["original_text"] == "Привет мир"
        assert len(data["paraphrases"]) > 0
        print(f"Simple text paraphrase: {data['paraphrases']}")

    @staticmethod
    def test_paraphrase_longer_text():
        """Тест парафраза более длинного текста"""
        payload = {
            "text": "Искусственный интеллект становится все более важной частью нашей жизни",
            "num_return_sequences": 2,
            "num_beams": 5,
            "temperature": 0.8
        }
        response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["paraphrases"]) == 2
        print("Longer text paraphrase:")
        for i, p in enumerate(data["paraphrases"], 1):
            print(f"  {i}. {p}")

    @staticmethod
    def test_paraphrase_multiple_sentences():
        """Тест парафраза нескольких предложений"""
        payload = {
            "text": "Машинное обучение это подраздел искусственного интеллекта. Оно позволяет компьютерам учиться на данных.",
            "num_return_sequences": 3,
            "num_beams": 5,
            "temperature": 1.0
        }
        response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["paraphrases"]) == 3
        print("Multiple sentences paraphrase:")
        for i, p in enumerate(data["paraphrases"], 1):
            print(f"  {i}. {p}")

    @staticmethod
    def test_paraphrase_technical_text():
        """Тест парафраза технического текста"""
        payload = {
            "text": "FastAPI это современный веб-фреймворк для создания REST API на Python",
            "num_return_sequences": 2,
            "num_beams": 5,
            "temperature": 0.7
        }
        response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["paraphrases"]) > 0
        print(f"Technical text paraphrase: {data['paraphrases']}")

    @staticmethod
    def test_paraphrase_with_different_temperature():
        """Тест парафраза с разными температурами"""
        text = "Тестирование это важная часть разработки программного обеспечения"
        temperatures = [0.5, 1.0, 1.5]

        print("Paraphrase with different temperatures:")
        for temp in temperatures:
            payload = {
                "text": text,
                "num_return_sequences": 1,
                "num_beams": 5,
                "temperature": temp
            }
            response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
            assert response.status_code == 200
            data = response.json()
            print(f"  Temperature {temp}: {data['paraphrases'][0]}")

    @staticmethod
    def test_paraphrase_with_different_beams():
        """Тест парафраза с разными значениями beam search"""
        text = "Разработка тестов это критически важный процесс"
        beams = [1, 3, 5, 10]

        print("Paraphrase with different beam widths:")
        for beam in beams:
            payload = {
                "text": text,
                "num_return_sequences": 1,
                "num_beams": beam,
                "temperature": 1.0
            }
            response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
            assert response.status_code == 200
            data = response.json()
            print(f"  Beams {beam}: {data['paraphrases'][0]}")

    @staticmethod
    def test_paraphrase_empty_text_error():
        """Тест на ошибку при пустом тексте"""
        payload = {"text": ""}
        response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
        assert response.status_code == 400
        print("Empty text error caught correctly")

    @staticmethod
    def test_paraphrase_whitespace_only_error():
        """Тест на ошибку при тексте только с пробелами"""
        payload = {"text": "   "}
        response = requests.post(f"{BASE_URL}/paraphrase", json=payload)
        assert response.status_code == 400
        print("Whitespace-only text error caught correctly")

    @staticmethod
    def test_device_info():
        """Проверка информации о device (CPU/CUDA)"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        device = data["device"]
        cuda_available = data["cuda_available"]
        print(f"Device: {device}, CUDA available: {cuda_available}")


def run_all_real_tests():
    """Запустить все тесты"""
    print("=" * 60)
    print("Запуск тестов API с моделью")
    print("=" * 60)

    test_instance = TestRealAPIRequests()

    tests = [
        ("Health check", test_instance.test_health_check),
        ("Simple text paraphrase", test_instance.test_paraphrase_simple_text),
        ("Longer text paraphrase", test_instance.test_paraphrase_longer_text),
        ("Multiple sentences", test_instance.test_paraphrase_multiple_sentences),
        ("Technical text", test_instance.test_paraphrase_technical_text),
        ("Different temperatures",
         test_instance.test_paraphrase_with_different_temperature),
        ("Different beam widths", test_instance.test_paraphrase_with_different_beams),
        ("Empty text error", test_instance.test_paraphrase_empty_text_error),
        ("Whitespace error", test_instance.test_paraphrase_whitespace_only_error),
        ("Device info", test_instance.test_device_info),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            test_func()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Результаты: {passed} passed,  {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_all_real_tests()
    except requests.exceptions.ConnectionError:
        print("Ошибка подключения! Убедись что сервер запущен на http://localhost:8000")
