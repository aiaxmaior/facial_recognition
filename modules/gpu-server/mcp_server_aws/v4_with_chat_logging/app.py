import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from config import Settings
from custom_logger import LogHandler
from data_model import ChatCompletionRequest
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from helper import generate_response, generate_stream_response
from logger_html import html_content
from loguru import logger
from mcp_client import MCPClient

load_dotenv()

settings = Settings()

# In-memory log storage
log_storage = deque(maxlen=1000)  # Store last 1000 log entries
websocket_connections = []


# Add custom handler to logger
log_handler = LogHandler(
    log_queue=log_storage, websocket_connections=websocket_connections
)
logger.add(log_handler.write, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


logger.info("Creating global MCPClient instance")
# client = MCPClient()


async def get_mcp_client(request: ChatCompletionRequest):
    """Dependency that provides MCPClient with automatic cleanup"""
    client = MCPClient()
    try:
        if request.mcp_servers:
            await client.connect_to_mcp_servers(request.mcp_servers)
            logger.info(f"Connected to {len(request.mcp_servers)} MCP servers")
        else:
            logger.info("No MCP servers provided in request, proceeding without tools")
        yield client
    except Exception as e:
        logger.error(f"Error connecting to MCP servers: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Error connecting to MCP servers: {str(e)}",
        )
    finally:
        try:
            await client.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up MCPClient: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application lifespan")
    try:
        # await client.connect_to_server(
        #     os.environ["MCP_SERVER_URL"]
        # )  # Update with actual path
        logger.info("FastAPI application startup completed")
        yield
    except Exception as e:
        logger.error(f"Error connecting to MCP server during startup: {str(e)}")
        print(f"Error connecting to MCP server: {str(e)}")
        yield
    finally:
        logger.info("Shutting down FastAPI application")
        # await client.cleanup()


logger.info("Creating FastAPI application")
app = FastAPI(lifespan=lifespan, root_path="/galaxy/slmapi")


@app.get("/")
async def get_default():
    logger.info("Health check endpoint called")
    return "MCP Client Application is running."


@app.get("/logs")
async def get_logs(limit: int = 100):
    """API endpoint to retrieve logs"""
    logger.info(f"Logs endpoint called with limit={limit}")
    logs = list(log_storage)
    return {"logs": logs[-limit:], "total": len(logs)}


@app.get("/logs/view", response_class=HTMLResponse)
async def view_logs():
    """Web UI to view logs in real-time"""
    return HTMLResponse(content=html_content)


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    websocket_connections.append(websocket)
    logger.info(
        f"WebSocket client connected. Total connections: {len(websocket_connections)}"
    )

    try:
        while True:
            # Keep the connection alive and wait for messages (if any)
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        logger.info(
            f"WebSocket connection closed. Remaining connections: {len(websocket_connections)}"
        )


@app.post("/v1/chat/completions")
@app.post("/v1/chat-html/completions")
async def get_response(
    request: ChatCompletionRequest,
    client: MCPClient = Depends(get_mcp_client),
):
    # streaming response
    try:
        response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
        created_timestamp = int(time.time())

        logger.info(
            f"Received chat completion request - ID: {response_id}, model: {request.model}, stream: {request.stream}"
        )
        logger.debug(
            f"Request details - messages: {len(request.messages)}, max_tokens: {request.max_tokens}, temperature: {request.temperature}"
        )

        if request.stream:
            logger.info(f"Returning streaming response for request ID: {response_id}")
            return StreamingResponse(
                generate_stream_response(
                    request,
                    client,
                    response_id,
                    created_timestamp,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming response
        logger.info(f"Generating non-streaming response for request ID: {response_id}")
        response = await generate_response(request, client)
        logger.info(f"Successfully returned response for request ID: {response_id}")
        return response
    except Exception as e:
        if request.stream:
            logger.error(
                f"Error generating streaming response for request ID {response_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Error generating streaming response for request ID {response_id}: {str(e)}",
            )
        else:
            logger.error(f"Error processing request ID {response_id}: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"Error processing request ID {response_id}: {str(e)}",
            )


# @app.post("/v1/chat-html/completions")
async def get_html_response(request: ChatCompletionRequest):
    # streaming response
    response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
    created_timestamp = int(time.time())

    logger.info(
        f"Received chat completion request - ID: {response_id}, model: {request.model}, stream: {request.stream}"
    )
    logger.debug(
        f"Request details - messages: {len(request.messages)}, max_tokens: {request.max_tokens}, temperature: {request.temperature}"
    )

    initial_response = await generate_response(request, client)

    initial_content = initial_response.choices[0].message.content

    html_msg_request = ChatCompletionRequest(
        model=request.model,
        messages=[
            {
                "role": "system",
                "content": """As an expert HTML formatter, your task is to convert user-provided text into well-structured, semantic HTML. 
        Use suitable tags to enhance the visual appeal and readability.
        Process the content by:
            - Identifying different sections (like paragraphs, lists, if required) within the text.
            - For heading use <h6>.
            - For new line use linebreak <br>. Strictly, do not use \n or \n\n.
            - [IMPORTANT] DO NOT use header tags like <h1>, <h2>, etc.
            - Applying the appropriate HTML tags to each section to improve semantics.
            - Generating the HTML structure and returning only the HTML content without commentary.
            - Ensuring no modifications to the original text aside from HTML formatting.
            - Upon receiving user content, apply these instructions meticulously.
            - Remove unwanted spaces, newlines or lines which are not required.
            - Do not include '\n\n' and '<br>\n' in your response.
            - Avoid unnecessary blank lines, line breaks and extra newlines in the HTML output; format it cleanly and compactly.""",
            },
            {
                "role": "user",
                "content": f"Format the following text into HTML:\n{initial_content}",
            },
        ],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream,
    )

    if request.stream:
        logger.info(f"Returning streaming response for request ID: {response_id}")
        return StreamingResponse(
            generate_stream_response(
                html_msg_request,
                client,
                response_id,
                created_timestamp,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming response
    try:
        logger.info(f"Generating non-streaming response for request ID: {response_id}")
        response = await generate_response(html_msg_request, client)
        logger.info(f"Successfully returned response for request ID: {response_id}")
        # replace \n with empty string
        new_content = response.choices[0].message.content.replace("\n", "")
        response.choices[0].message.content = new_content
        return response
    except Exception as e:
        logger.error(f"Error processing request ID {response_id}: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # logger.info(
    #     f"Starting uvicorn server on host 0.0.0.0, port {settings['client_port']}"
    # )
    # uvicorn.run(app, host="0.0.0.0", port=settings["client_port"])
    # uvicorn.run(app, host="0.0.0.0", port=10010)
    uvicorn.run(app, host="0.0.0.0", port=3000)