# import os

# import requests
# from dotenv import load_dotenv
# from pymongo import MongoClient

# load_dotenv()


# def fetch_model_name():
#     response = requests.get(
#         os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
#     )
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch model name: {response.status_code}")
#     data = response.json()
#     bridge_url = data["data"]["connectionString"]
#     # create a pymongo client to fetch dlmObj
#     client = MongoClient(bridge_url)
#     db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
#     collection = db["bridgeMetaInfo"]
#     dlmObj = collection.find_one()["dlmObj"]["hrDlm"]
#     model_name = dlmObj["modelName"]
#     return model_name


# ### create same tool as fetch_model_name but for llm_base_url
# def fetch_llm_base_url():
#     response = requests.get(
#         os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
#     )
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch llm base url: {response.status_code}")
#     data = response.json()
#     bridge_url = data["data"]["connectionString"]
#     # create a pymongo client to fetch dlmObj
#     client = MongoClient(bridge_url)
#     db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
#     collection = db["bridgeMetaInfo"]
#     dlmObj = collection.find_one()
#     dlmObj = dlmObj["dlmObj"]["hrDlm"]
#     llm_base_url = dlmObj["llmBaseUrl"]
#     return llm_base_url


# def fetch_mcp_server_url():
#     response = requests.get(
#         os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
#     )
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch mcp server url: {response.status_code}")
#     data = response.json()
#     bridge_url = data["data"]["connectionString"]
#     client = MongoClient(bridge_url)
#     db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
#     collection = db["bridgeMetaInfo"]
#     dlmObj = collection.find_one()
#     dlmObj = dlmObj["dlmObj"]["hrDlm"]
#     mcp_server_url = dlmObj["mcpServerUrl"]
#     return mcp_server_url


# def fetch_hr_assistant_endpoint():
#     response = requests.get(
#         os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
#     )
#     if response.status_code != 200:
#         raise Exception(
#             f"Failed to fetch hr assistant endpoint: {response.status_code}"
#         )
#     data = response.json()
#     bridge_url = data["data"]["connectionString"]
#     client = MongoClient(bridge_url)
#     db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
#     collection = db["bridgeMetaInfo"]
#     dlmObj = collection.find_one()
#     dlmObj = dlmObj["dlmObj"]["hrDlm"]
#     rag_endpoint = dlmObj["ragEndpoint"]
#     return rag_endpoint

import os

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def fetch_model_name():
    # response = requests.get(
    #     os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
    # )
    # if response.status_code != 200:
    #     raise Exception(f"Failed to fetch model name: {response.status_code}")
    # data = response.json()
    # bridge_url = data["data"]["connectionString"]
    # # create a pymongo client to fetch dlmObj
    # client = MongoClient(bridge_url)
    # db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
    # collection = db["bridgeMetaInfo"]
    # dlmObj = collection.find_one()["dlmObj"]["hrDlm"]
    # model_name = dlmObj["modelName"]
    # #return model_name
    return "Qwen/Qwen3-4B-Instruct-2507"

### create same tool as fetch_model_name but for llm_base_url
def fetch_llm_base_url():
    # response = requests.get(
    #     os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
    # )
    # if response.status_code != 200:
    #     raise Exception(f"Failed to fetch llm base url: {response.status_code}")
    # data = response.json()
    # bridge_url = data["data"]["connectionString"]
    # # create a pymongo client to fetch dlmObj
    # client = MongoClient(bridge_url)
    # db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
    # collection = db["bridgeMetaInfo"]
    # dlmObj = collection.find_one()
    # dlmObj = dlmObj["dlmObj"]["hrDlm"]
    # llm_base_url = dlmObj["llmBaseUrl"]
    # return llm_base_url
    return "https://mcp-llm-client.qryde.net/v1"


def fetch_mcp_server_url():
    response = requests.get(
        os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
    )
    if response.status_code != 200:
        raise Exception(f"Failed to fetch mcp server url: {response.status_code}")
    data = response.json()
    bridge_url = data["data"]["connectionString"]
    client = MongoClient(bridge_url)
    db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
    collection = db["bridgeMetaInfo"]
    dlmObj = collection.find_one()
    dlmObj = dlmObj["dlmObj"]["hrDlm"]
    mcp_server_url = dlmObj["mcpServerUrl"]
    return mcp_server_url


def fetch_hr_assistant_endpoint():
    # response = requests.get(
    #     os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
    # )
    # if response.status_code != 200:
    #     raise Exception(
    #         f"Failed to fetch hr assistant endpoint: {response.status_code}"
    #     )
    # data = response.json()
    # bridge_url = data["data"]["connectionString"]
    # client = MongoClient(bridge_url)
    # db = client[f"{os.environ['TENANT_ID'].lower()}-bridge"]
    # collection = db["bridgeMetaInfo"]
    # dlmObj = collection.find_one()
    # dlmObj = dlmObj["dlmObj"]["hrDlm"]
    # rag_endpoint = dlmObj["ragEndpoint"]
    # return rag_endpoint
    return "https://slm-kb.qraie.ai/api/get_chunks"
