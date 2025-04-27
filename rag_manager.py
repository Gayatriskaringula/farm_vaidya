import os
import shutil
import asyncio
from config import WORKING_DIR
from gemini_llm import llm_model_func
from embeddings import embedding_func
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

class RagManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.rag = None
        asyncio.run(self._initialize_rag())
        asyncio.run(self._insert_text())

    async def _initialize_rag(self):
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        os.mkdir(WORKING_DIR)

        self.rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    async def _insert_text(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            text = file.read()
        self.rag.insert(text)
        print(" Dickens text inserted into RAG!")

    def query(self, question: str):
        result = asyncio.run(self.rag.query(QueryParam(query=question)))
        return result.answer
