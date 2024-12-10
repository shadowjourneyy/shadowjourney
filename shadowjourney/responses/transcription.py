from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Word:
    word: str
    start: float
    end: float

@dataclass
class Segment:
    id: int
    seek: float
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

@dataclass
class TranscriptionResponse:
    text: str
    language: Optional[str] = None
    duration: Optional[str] = None
    words: Optional[List[Word]] = None
    segments: Optional[List[Segment]] = None
