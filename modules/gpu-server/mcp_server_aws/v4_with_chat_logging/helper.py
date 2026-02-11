import json
import time
import uuid
from typing import AsyncGenerator

from data_model import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from loguru import logger
from mcp_client import MCPClient


async def generate_stream_response(
    request: ChatCompletionRequest,
    client: MCPClient,
    response_id: str,
    created_timestamp: int,
) -> AsyncGenerator[str, None]:
    """Generate streaming response in OpenAI format"""
    logger.info(f"Generating stream response for request ID: {response_id}")
    try:
        chunk_count = 0
        async for content_chunk in client.process_query_stream(request):
            chunk_count += 1
            chunk_response = ChatCompletionStreamResponse(
                id=response_id,
                created=created_timestamp,
                model=request.model,
                object="chat.completion.chunk",
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatMessage(role="assistant", content=content_chunk),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk_response.model_dump_json()}\n\n"

        logger.info(
            f"Stream response generated {chunk_count} chunks for request ID: {response_id}"
        )

        final_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created_timestamp,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            # TO DO fix this - calculate tokens correctly
            usage=ChatCompletionUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        ### SSE's specification
        yield "data: [DONE]\n\n"
        logger.debug(f"Stream response completed for request ID: {response_id}")
    except Exception as e:
        logger.error(
            f"Error in stream response generation for request ID {response_id}: {str(e)}"
        )
        error_chunk = {"error": str(e)}
        # yield f"data: {json.dumps(error_chunk)}\n\n"
        raise


async def generate_response(
    request: ChatCompletionRequest,
    client: MCPClient,
) -> ChatCompletionResponse:
    """Generate response in OpenAI format"""
    response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
    logger.info(f"Generating response for request ID: {response_id}")
    logger.debug(
        f"Request model: {request.model}, messages count: {len(request.messages)}"
    )

    try:
        response_content = await client.process_query(request)
        logger.info(f"Response content generated for request ID: {response_id}")
    except Exception as e:
        logger.error(
            f"Failed to fetch response from LLM for request ID {response_id}: {str(e)}"
        )
        raise

    # Create OpenAI-compatible response
    created_timestamp = int(time.time())

    # Estimate token usage (simple approximation) - wrong
    # TO DO
    try:
        # prompt_tokens = sum(
        #     len(getattr(msg, "content").split()) for msg in request.messages
        # )
        prompt_tokens = 0
        completion_tokens = len(response_content.split())
        total_tokens = prompt_tokens + completion_tokens
        logger.debug(
            f"Token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}"
        )
    except Exception as e:
        logger.error(
            f"Failed to calculate usage tokens for request ID {response_id}: {str(e)}"
        )
        raise

    response = ChatCompletionResponse(
        id=response_id,
        created=created_timestamp,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )

    logger.info(f"Response generated successfully for request ID: {response_id}")
    return response
