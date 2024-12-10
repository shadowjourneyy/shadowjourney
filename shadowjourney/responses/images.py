from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Image:
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    url: str = ""

@dataclass
class ImagesResponse:
    created: int
    data: List[Image]