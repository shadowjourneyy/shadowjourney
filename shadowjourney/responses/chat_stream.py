from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ChoiceDelta:
    content: Optional[str] = None
    function_call: Optional[dict] = None
    refusal: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[dict] = None

@dataclass
class Choice:
    delta: ChoiceDelta
    finish_reason: Optional[str]
    index: int
    logprobs: Optional[dict] = None

@dataclass
class ChatCompletionChunk:
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[dict] = None
