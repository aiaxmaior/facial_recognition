# import os

# import requests
# from dotenv import load_dotenv
# from pymongo import MongoClient

# load_dotenv()


# def fetch_wfm_uri():
#     response = requests.get(
#         os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
#     )
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch llm base url: {response.status_code}")
#     data = response.json()
#     bridge_url = data["data"]["connectionString"]
#     return bridge_url


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


def fetch_wfm_uri():
    # response = requests.get(
    #     os.environ["CONF_URL"] + "/" + os.environ["TENANT_ID"], verify=False
    # )
    # if response.status_code != 200:
    #     raise Exception(f"Failed to fetch llm base url: {response.status_code}")
    # data = response.json()
    # bridge_url = data["data"]["connectionString"]
    # return bridge_url
    return "mongodb+srv://shreenivas1003joshi_db_user:2VzwUNI1E0enzDhu@cluster0.qgmw3m0.mongodb.net/?appName=Cluster0"


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
    return "https://slm-kb.qraie.ai/api/get_chunks"
