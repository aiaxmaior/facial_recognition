import json
import os
import re
from typing import Optional

from fetch_config import fetch_hr_assistant_endpoint
from httpx import AsyncClient
from loguru import logger


def extract_eRep_id(messages: list) -> Optional[str]:
    pattern = r"eRep Id:\s*([^ ,]*)"
    for msg in messages:
        if msg.role == "system" and msg.content:
            match = re.search(pattern, msg.content)
            if match:
                eRep_id = match.group(1).strip()
                logger.info(f"Extracted eRep_id: {eRep_id}")
                return eRep_id
    logger.warning("No eRep_id found in messages")
    return None


async def rag_assistant(question: str, eRep_id: str) -> Optional[str]:
    tenant_id = os.environ["TENANT_ID"].lower()
    client = AsyncClient()
    try:
        response = await client.post(
            fetch_hr_assistant_endpoint(),
            json={
                "eRep_id": eRep_id,
                "tenant_id": tenant_id,
                "question": question,
                "chat_history": [],
            },
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJicmlkZ2VVc2VyIiwiZXhwIjoxNzk3NjY2ODM0fQ.gF6TbvnPa2CCMAV-a67Ec2V3Z4pg9ORyGHnpldcpkBs"},
            timeout=100.0,
        )

        if response.status_code == 200:
            context_docs = response.json().get("context", [])
            if context_docs:
                retrieved_docs = "\n\nDocument: ".join(
                    [doc["page_content"] for doc in context_docs]
                )
                return f"Retrieved documents: {retrieved_docs}"
            else:
                return None
    except Exception as e:
        logger.error(f"Error calling RAG assistant: {str(e)}")
        return None
    finally:
        await client.aclose()
