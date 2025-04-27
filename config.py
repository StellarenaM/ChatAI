from dataclasses import dataclass


@dataclass
class OllamaConfig:
    API_URL = "http://localhost:11435/v1"
    API_KEY = "pass"
    Qwen_MODEL_NAME = 'qwen2:7b-instruct'
    DeepSeek_MODEL_NAME = 'deepseek-r1:7b'
    USE_SPECIAL_PROMPT = True
