import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, AsyncGenerator, Dict, List, Optional

from client_helper import extract_eRep_id, rag_assistant
from config import Settings
from data_model import ChatCompletionRequest
from fetch_config import fetch_llm_base_url, fetch_model_name
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import AsyncOpenAI

settings = Settings()


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        logger.info("Initializing MCPClient")
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_server: Dict[str, str] = {}
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.openai = AsyncOpenAI(
            base_url=fetch_llm_base_url(),
            api_key="ollama",
        )
        logger.info("MCPClient initialized successfully")

    # methods will go here

    async def connect_to_server(self, server_http_url: str):
        """Connect to an MCP server

        Args:
            server_http_url: URL of the MCP server
        """
        if server_http_url in self.sessions:
            logger.info(f"MCP server at {server_http_url} already connected")
            return

        logger.info(f"Connecting to mcp server at: {server_http_url}")
        try:
            streamable_transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(server_http_url)
            )
            read_stream, write_stream, _ = streamable_transport
            logger.debug("Streamable transport established")

            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            logger.info(f"Connected to mcp server at: {server_http_url}")

            await session.initialize()
            logger.debug("Session initialized")

            # List available tools
            response = await session.list_tools()
            tools = response.tools

            # store this server's session
            self.sessions[server_http_url] = session

            # Map tools to their server for routing
            for tool in tools:
                if tool.name in self.tool_to_server:
                    logger.warning(
                        f"Tool '{tool.name}' already mapped to server {self.tool_to_server[tool.name]}"
                    )
                else:
                    self.tool_to_server[tool.name] = server_http_url
                    logger.info(
                        f"Mapped tool '{tool.name}' to server {server_http_url}"
                    )

            logger.info(
                f"Session initialized with mcp server at {server_http_url} with {len(tools)} tools: {[tool.name for tool in tools]}"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to MCP server at {server_http_url}: {str(e)}"
            )
            raise

    async def connect_to_mcp_servers(self, server_http_urls: List[str]):
        """Connect to multiple MCP servers"""
        logger.info(f"Connecting to {len(server_http_urls)} MCP servers")
        # tasks = [
        #     self.connect_to_server(server_http_url)
        #     for server_http_url in server_http_urls
        # ]
        # await asyncio.gather(*tasks)
        for server_http_url in server_http_urls:
            await self.connect_to_server(server_http_url)
        logger.info(
            f"Connected to all {len(server_http_urls)} MCP servers successfully"
        )

    async def process_query(self, request: ChatCompletionRequest) -> str:
        """Process a query using OpenAI and available tools"""
        messages = request.messages
        logger.info(f"Processing query with {len(messages)} messages")
        logger.debug(f"Query messages: {messages}")

        # Extract last user query and fetch retrieval chunks
        user_query = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_query = msg.content
                break

        if user_query:
            logger.info(f"User query: {user_query}")
            # eRep_id = extract_eRep_id(messages)
            if request.eRep_id:
                eRep_id = request.eRep_id
            else:
                eRep_id = extract_eRep_id(messages)

            if eRep_id:
                logger.info(f"Fetching retrieval chunks for eRep_id: {eRep_id}")
                retreived_context = await rag_assistant(user_query, eRep_id)
                if retreived_context:
                    logger.info(
                        f"Retrieved context: {retreived_context}, adding to messages"
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": retreived_context,
                        }
                    )
                else:
                    logger.info("No retrieval chunks found")
            else:
                logger.warning("No eRep_id found, continuing without retrieval")

        available_tools = await self.get_all_tools()

        logger.info(
            f"Available tools from {len(self.sessions)} MCP servers: {[tool['function']['name'] for tool in available_tools]}"
        )

        logger.debug("Calling OpenAI chat completions API")
        if available_tools:
            response = await self.openai.chat.completions.create(
                model=fetch_model_name(),
                messages=messages,
                tools=available_tools,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            response = await self.openai.chat.completions.create(
                model=fetch_model_name(),
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        logger.info(f"Response generated from OpenAI")
        # Process response and handle tool calls
        final_text = []
        message = response.choices[0].message
        logger.debug(f"Message: {message}")
        # Check if the response contains tool calls
        if message.tool_calls:
            logger.info(f"Processing {len(message.tool_calls)} tool call(s)")
            # Handle tool calls
            tool_results = []
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(
                    tool_call.function.arguments
                )  # Parse JSON string to dict

                logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
                # Execute tool call
                try:
                    result = await self.call_tool(tool_name, tool_args)
                    logger.info(f"Tool '{tool_name}' execution successful")
                    logger.debug(f"Tool '{tool_name}' result: {result}")
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' execution failed: {str(e)}")
                    raise
                tool_results.append((tool_call.id, result))

            # Add one assistant message with all tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls,
                }
            )

            # Add each tool result
            for tool_call_id, result in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": str(result.content),
                    }
                )
            logger.debug(f"Updated messages with tool results")

            # Get next response
            logger.debug("Getting next response after all tool execution")
            if available_tools:
                response = await self.openai.chat.completions.create(
                    model=fetch_model_name(),
                    messages=messages,
                    tools=available_tools,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    presence_penalty=request.presence_penalty,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            else:
                response = await self.openai.chat.completions.create(
                    model=fetch_model_name(),
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    presence_penalty=request.presence_penalty,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

            logger.info(f"Response generated from OpenAI")
            # Add the final response content
            final_text.append(response.choices[0].message.content)
            logger.debug("Appended final response content")
        else:
            # No tool calls, just return the content
            logger.info("No tool calls detected, returning direct response")
            final_text.append(message.content)

        logger.info(
            f"Query processing complete with {len(final_text)} response part(s)"
        )
        logger.debug(f"Final response: {final_text}")
        return "\n".join(final_text)

    async def process_query_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        messages = request.messages
        logger.info(f"Processing streaming query with {len(messages)} messages")
        logger.debug(f"Streaming query messages: {messages}")

        # accumulator for logging the complete streamed response
        accumulated_response = []

        # Extract last user query and fetch retrieval chunks
        user_query = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_query = msg.content
                break
        if user_query:
            logger.info(f"User query: {user_query}")
            # eRep_id = extract_eRep_id(messages)
            if request.eRep_id:
                eRep_id = request.eRep_id
            else:
                eRep_id = extract_eRep_id(messages)
            if eRep_id:
                logger.info(f"Fetching retrieval chunks for eRep_id: {eRep_id}")
                retreived_context = await rag_assistant(user_query, eRep_id)
                if retreived_context:
                    logger.info(
                        f"Retrieved context: {retreived_context}, adding to messages"
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": retreived_context,
                        }
                    )
                else:
                    logger.info("No retrieval chunks found")
            else:
                logger.warning("No eRep_id found, continuing without retrieval")

        available_tools = await self.get_all_tools()

        logger.info(
            f"Available tools for streaming from {len(self.sessions)} MCP servers: {[tool['function']['name'] for tool in available_tools]}"
        )
        logger.debug("Creating streaming chat completion")
        if available_tools:
            stream = await self.openai.chat.completions.create(
                model=fetch_model_name(),
                messages=messages,
                tools=available_tools,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=True,
            )
        else:
            stream = await self.openai.chat.completions.create(
                model=fetch_model_name(),
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                presence_penalty=request.presence_penalty,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                stream=True,
            )
        logger.info(f"Streaming chat completion created")

        # to handle multiple tool calls, storing them first
        current_tool_calls = {}
        logger.debug("Starting to process streaming chunks")

        async for chunk in stream:
            choice = chunk.choices[0]

            # text streaming
            if choice.delta.content is not None:
                accumulated_response.append(choice.delta.content)
                yield choice.delta.content

            # tool streaming
            if choice.delta.tool_calls:
                for tool_call_chunk in choice.delta.tool_calls:
                    # yield f"\n[Tool call chunk that we got {tool_call_chunk}]"
                    # Use index as fallback if no ID is provided
                    # tool_call_id = tool_call_chunk.id or str(tool_call_chunk.index or 0)
                    if tool_call_chunk.id:
                        tool_call_id = tool_call_chunk.id
                    else:
                        # use the same id which we got in last chunk
                        # not sure if this would work for parallel tool calls
                        pass

                    if tool_call_id not in current_tool_calls:
                        current_tool_calls[tool_call_id] = {
                            "id": tool_call_chunk.id or tool_call_id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    # add up function name
                    if tool_call_chunk.function.name:  # function_name:
                        current_tool_calls[tool_call_id]["function"][
                            "name"
                        ] += tool_call_chunk.function.name

                    # add up tool call arguments
                    if tool_call_chunk.function.arguments:
                        current_tool_calls[tool_call_id]["function"][
                            "arguments"
                        ] += tool_call_chunk.function.arguments

                # yield f"\n[Tool call that we parsed {current_tool_calls}]"

            # check if end of tool_call or stream
            if choice.finish_reason == "tool_calls":
                logger.info(
                    f"Finish reason: tool_calls, processing {len(current_tool_calls)} tool call(s)"
                )

                # collect all results
                tool_results = []
                # collect all tool call
                openai_tool_calls = []
                # Execute all tool calls
                for tool_call_id, tool_call in current_tool_calls.items():
                    try:
                        tool_name = tool_call["function"]["name"]
                        tool_args_str = tool_call["function"]["arguments"]
                        tool_args = json.loads(tool_args_str)

                        logger.info(
                            f"Executing streaming tool '{tool_name}' with args: {tool_args}"
                        )
                        result = await self.call_tool(tool_name, tool_args)
                        logger.info(
                            f"Streaming tool '{tool_name}' execution successful"
                        )
                        logger.debug(f"Streaming tool '{tool_name}' result: {result}")

                        # Convert our internal tool_call format to OpenAI format
                        openai_tool_call = {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }

                        openai_tool_calls.append(openai_tool_call)
                        tool_results.append((tool_call["id"], result))

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Error parsing tool arguments for {tool_name}: {str(e)}"
                        )
                        # yield f"\n[Error parsing tool arguments for {tool_name}: {str(e)}]\n"
                        raise
                    except Exception as e:
                        logger.error(
                            f"Error executing streaming tool {tool_name}: {str(e)}"
                        )
                        # yield f"\n[Error executing tool {tool_name}: {str(e)}]\n"
                        raise

                # add one assistant message with all tool calls
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": openai_tool_calls,
                    }
                )
                logger.debug("Updated messages with all streaming tool calls")

                for tool_call_id, result in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": str(result.content),
                        }
                    )
                logger.debug("Updated messages with streaming tool results")

                # Get final response after tool execution
                logger.debug("Getting final streaming response after tool execution")
                if available_tools:
                    next_stream = await self.openai.chat.completions.create(
                        model=fetch_model_name(),
                        messages=messages,
                        tools=available_tools,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        presence_penalty=request.presence_penalty,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        stream=True,
                    )
                else:
                    next_stream = await self.openai.chat.completions.create(
                        model=fetch_model_name(),
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        presence_penalty=request.presence_penalty,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        stream=True,
                    )
                logger.info(f"Final streaming response created")

                # Stream the final response
                async for next_chunk in next_stream:
                    if next_chunk.choices[0].delta.content is not None:
                        accumulated_response.append(next_chunk.choices[0].delta.content)
                        yield next_chunk.choices[0].delta.content

                    # Handle potential nested tool calls (though less common)
                    if next_chunk.choices[0].finish_reason in ["stop", "length"]:
                        logger.info(
                            f"Streaming complete with finish_reason: {next_chunk.choices[0].finish_reason}"
                        )
                        break

                break
        complete_response = "".join(accumulated_response)
        logger.info(
            f"Complete streamed response ({len(accumulated_response)} chunks): {complete_response}"
        )

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected MCP servers"""
        all_tools = []
        for server_http_url, session in self.sessions.items():
            try:
                response = await session.list_tools()
                for tool in response.tools:
                    all_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Error getting tools from MCP server at {server_http_url}: {str(e)}"
                )
        return all_tools

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]):
        """Route tool call to the appropriate MCP server"""
        if tool_name not in self.tool_to_server:
            logger.warning(f"Tool '{tool_name}' not found.")
            raise ValueError(f"Tool '{tool_name}' not found.")
        server_http_url = self.tool_to_server[tool_name]
        session = self.sessions[server_http_url]

        logger.info(f"Calling tool {tool_name} on server {server_http_url}")
        result = await session.call_tool(tool_name, tool_args)
        return result

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Starting cleanup process")

        # First, close the OpenAI client
        try:
            if self.openai:
                logger.debug("Closing AsyncOpenAI client...")
                await self.openai.close()
                logger.info("AsyncOpenAI client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing AsyncOpenAI client (non-critical): {str(e)}")

        # Then close the exit stack (which includes MCP session and transport)
        try:
            logger.debug("Closing exit stack (MCP session and transport)...")
            await self.exit_stack.aclose()
            logger.info("Exit stack closed successfully")
        except BaseExceptionGroup as eg:  # Handle exception groups from TaskGroup
            logger.warning(f"Exception group during exit stack cleanup: {eg}")
            for exc in eg.exceptions:
                logger.warning(f"  - Sub-exception: {type(exc).__name__}: {exc}")
        except Exception as e:
            logger.warning(f"Error during exit stack cleanup (non-critical): {str(e)}")

        logger.info("Cleanup process completed")
        # try:
        #     # Close OpenAI client first
        #     if self.openai:
        #         await self.openai.close()
        #         logger.info("AsyncOpenAI client closed successfully")
        #     await self.exit_stack.aclose()
        #     logger.info("Cleanup completed successfully")
        # except Exception as e:
        #     logger.error(f"Error during cleanup: {str(e)}")
        #     raise
