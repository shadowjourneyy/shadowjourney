from dataclasses import dataclass
from typing import Optional, List

@dataclass
class CompletionTokensDetails:
    accepted_prediction_tokens: int
    audio_tokens: Optional[int]
    reasoning_tokens: int
    rejected_prediction_tokens: int

@dataclass
class PromptTokensDetails:
    audio_tokens: Optional[int]
    cached_tokens: int

@dataclass
class CompletionUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails

@dataclass
class ChatCompletionMessage:
    content: str
    refusal: Optional[str]
    role: str
    audio: Optional[str]
    function_call: Optional[dict]
    tool_calls: Optional[list]

@dataclass
class Choice:
    finish_reason: str
    index: int
    logprobs: Optional[dict]
    message: ChatCompletionMessage

@dataclass
class ChatCompletion:
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    service_tier: Optional[str]
    system_fingerprint: Optional[str]
    usage: CompletionUsage
