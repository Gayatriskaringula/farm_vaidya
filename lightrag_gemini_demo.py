import os
import numpy as np
import litellm
from litellm import acompletion
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio

# Apply to avoid event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()
litellm.api_key = "gemini api-key"  # Use your Gemini key
litellm.api_base = "http://localhost:4000"     # LiteLLM proxy URL
model_name = "gemini-1.5-flash"               # Alias from config.yaml

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil
    shutil.rmtree(WORKING_DIR)
os.mkdir(WORKING_DIR)

# LiteLLM-compatible LLM call
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages += history_messages or []
    messages.append({"role": "user", "content": prompt})

    response = await litellm.acompletion(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_numpy=True)

# RAG initializer
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# Main function
def main():
    rag = asyncio.run(initialize_rag())
    print(rag)

if __name__ == "__main__":
    main()
