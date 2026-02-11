from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str = Literal["developer", "system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2500
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    parallel_tool_calls: Optional[bool] = False
    response_format: Optional[Dict[str, Any]] = None
    eRep_id: Optional[str] = None
    tenant_id: Optional[str] = None
    mcp_servers: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# stream classes
class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    choices: List[ChatCompletionStreamChoice]
    id: str
    created: int
    model: str
    object: str = "chat.completion.chunk"
    usage: Optional[ChatCompletionUsage] = None
