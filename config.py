import os
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

WORKING_DIR = "./dickens"
JSON_FILE_PATH = r"D:\farm_vaidya\LightRAG\dickens\kv_store_llm_response_cache.json"
