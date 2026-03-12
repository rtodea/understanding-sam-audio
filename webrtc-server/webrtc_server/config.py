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

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def effective_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
