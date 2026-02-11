"""
Original Gateway Code (Formatted)
From Admin - Shows existing gateway pattern
"""

import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import markdown
import uvicorn
from custom_logger import LogHandler
from data_model import ChatCompletionRequest
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fetch_config import fetch_llm_base_url, fetch_model_name
from helper import generate_response, generate_stream_response
from logger_html import html_content
from loguru import logger
from mcp_client import MCPClient

load_dotenv()

settings = {
    "llm_base_url": fetch_llm_base_url(),
    "model_name": fetch_model_name(),
    "mcp_server_url": os.environ["MCP_SERVER_URL"],
}

# In-memory log storage
log_storage = deque(maxlen=1000)  # Store last 1000 log entries
websocket_connections = []

# Add custom handler to logger
log_handler = LogHandler(
    log_queue=log_storage,
    websocket_connections=websocket_connections
)
logger.add(
    log_handler.write,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

logger.info("Creating global MCPClient instance")
client = MCPClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application lifespan")
    try:
        await client.connect_to_server(
            settings["mcp_server_url"],
        )
        logger.info("FastAPI application startup completed")
        yield
    except Exception as e:
        logger.error(f"Error connecting to MCP server during startup: {str(e)}")
        print(f"Error connecting to MCP server: {str(e)}")
        yield
    finally:
        logger.info("Shutting down FastAPI application")
        await client.cleanup()


logger.info("Creating FastAPI application")

# ============================================
# KEY: Gateway integration via root_path
# ============================================
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
async def get_response(request: ChatCompletionRequest):
    """Chat completion endpoint (streaming + non-streaming)"""
    # streaming response
    try:
        response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
        created_timestamp = int(time.time())

        logger.info(
            f"Received chat completion request - ID: {response_id}, "
            f"model: {request.model}, stream: {request.stream}"
        )
        logger.debug(
            f"Request details - messages: {len(request.messages)}, "
            f"max_tokens: {request.max_tokens}, temperature: {request.temperature}"
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
    except Exception as e:
        logger.error(
            f"Error generating streaming response for request ID {response_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Error generating streaming response for request ID {response_id}: {str(e)}",
        )

    # Non-streaming response
    try:
        logger.info(f"Generating non-streaming response for request ID: {response_id}")
        response = await generate_response(request, client)
        logger.info(f"Successfully returned response for request ID: {response_id}")
        return response
    except Exception as e:
        logger.error(f"Error processing request ID {response_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Error processing request ID {response_id}: {str(e)}",
        )


if __name__ == "__main__":
    logger.info(
        f"Starting uvicorn server on host 0.0.0.0, port {settings['client_port']}"
    )
    uvicorn.run(app, host="0.0.0.0", port=settings["client_port"])
