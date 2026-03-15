import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sam_model: str = "facebook/sam-audio-large"
    sam_reranking_candidates: int = 1
    chunk_seconds: float = 3.0
    overlap_seconds: float = 1.5
    sample_rate: int = 48_000
    predict_spans: bool = False
    # Override device explicitly ("cuda" / "cpu"). Auto-detected when empty.
    device: str = ""

    # Azure Speech-to-Text (optional — STT is disabled when key is empty).
    azure_stt_key: str = ""
    azure_stt_region: str = "eastus"
    azure_stt_language: str = "en-US"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def stt_enabled(self) -> bool:
        return bool(self.azure_stt_key)

    @property
    def effective_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
