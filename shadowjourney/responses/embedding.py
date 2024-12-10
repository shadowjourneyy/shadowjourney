from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Embedding:
    embedding: List[float]
    index: int
    object: str = "embedding"

@dataclass
class Usage:
    prompt_tokens: int
    total_tokens: int

@dataclass
class CreateEmbeddingResponse:
    data: List[Embedding]
    model: Optional[str]
    object: str = "list"
    usage: Optional[Usage] = None