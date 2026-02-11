from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"  # comment this
    llm_base_url: str = "https://mcp-llm-client.qryde.net/v1"  # comment this
    mcp_server_url: str = "http://localhost:10011/mcp"  # internal can be hardcoded
    client_port: int = 5000  # internal can be hardcoded
