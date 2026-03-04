import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from utils import Utils
from typing import List, Dict, Any


class Startup:
    def __init__(self, index_path="data/faiss.index", meta_path="data/chunks.json"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        self.index_path = index_path
        self.meta_path = meta_path

        # Embeddings (all-MiniLM-L6-v2 => 384 dims)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.embedding_model.get_sentence_embedding_dimension()

        # Load / create FAISS index with IDs support
        self.index = self._load_or_create_index()

        # Load / create metadata store: vector_id -> {text, doc_id, ...}
        self.meta: Dict[str, Any] = self._load_or_create_meta()

        # Chat model
        self.chat_model = self.load_chat_model()

        # next vector id (monotonic) for chunk inserts
        self.next_vid = self._compute_next_vid()

    def load_chat_model(self):
        token = Utils().token
        endpoint = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=token,
            task="conversational",
        )
        return ChatHuggingFace(llm=endpoint)

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            idx = faiss.read_index(self.index_path)
            return idx

        base = faiss.IndexFlatL2(self.dim)
        idx = faiss.IndexIDMap2(base) 
        return idx

    def _load_or_create_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _compute_next_vid(self) -> int:
        if not self.meta:
            return 0
        # meta keys are strings
        return max(int(k) for k in self.meta.keys()) + 1

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

        return chunks

    def add_document_to_index(self, content: str, doc_id: str, source: str = "") -> int:
        chunks = self.chunk_text(content)
        if not chunks:
            return 0

        # Batch encode for speed; FAISS expects float32
        vecs = self.embedding_model.encode(chunks, convert_to_numpy=True)
        vecs = vecs.astype(np.float32)

        # Create unique vector ids (not doc_id). Store doc_id in metadata.
        vids = np.arange(self.next_vid, self.next_vid + len(chunks)).astype(np.int64)
        self.next_vid += len(chunks)

        # Add vectors with ids
        self.index.add_with_ids(vecs, vids)

        # Persist metadata for retrieval later
        for vid, chunk_text in zip(vids, chunks):
            self.meta[str(int(vid))] = {
                "doc_id": doc_id,
                "source": source,
                "text": chunk_text,
            }

        self.save()
        return len(chunks)

    def retrieve(self, query: str, k: int = 4):
        q = self.embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, ids = self.index.search(q, k)

        results = []
        for dist, vid in zip(distances[0], ids[0]):
            if vid == -1:
                continue
            item = self.meta.get(str(int(vid)))
            if item:
                results.append({"id": int(vid), "score": float(dist), **item})
        return results

    async def answer_with_rag(self, question: str, k: int = 4) -> str:
        hits = self.retrieve(question, k=k)
        context = "\n\n".join(
            [f"[doc:{h['doc_id']}] {h['text']}" for h in hits]
        ) or "No relevant context found."

        messages = [
            SystemMessage(content="Answer using the provided context. If insufficient, say so."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
        ]
        resp = await self.chat_model.ainvoke(messages)
        return resp.content