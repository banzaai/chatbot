# Chatbot with RAG (FAISS) - Usage Guide

Your chatbot now has **Retrieval-Augmented Generation (RAG)** with FAISS! Here's how to use it:

## Endpoints

### 1. Upload Document
Uploads a document and automatically indexes it into FAISS.

```bash
curl -X POST "http://localhost:8000/chat/upload_document" \
  -F "conversation_id=conv_123" \
  -F "file=@/path/to/document.txt"
```

**Response:**
```json
{
  "document_id": "uuid-here",
  "filename": "document.txt",
  "path": "data/uploads/uuid-here.txt",
  "status": "indexed"
}
```

### 2. Chat with RAG
Regular chat that automatically uses indexed documents.

```bash
curl -X POST "http://localhost:8000/chat/conv_123" \
  -H "Content-Type: application/json" \
  -d '{"messages": "What is in the uploaded document?"}'
```

The chatbot will:
- Search FAISS for relevant chunks
- Include them in the context
- Generate answers based on documents + conversation history

### 3. Query with RAG
Direct query against indexed documents with sources.

```bash
curl -X GET "http://localhost:8000/chat/query?conversation_id=conv_123&user_message=Tell%20me%20about%20X"
```

**Response:**
```json
{
  "response": "Answer based on documents...",
  "sources": [
    {
      "id": 0,
      "score": 0.123,
      "doc_id": "uuid",
      "source": "document.txt",
      "text": "Relevant chunk of text..."
    }
  ]
}
```

### 4. Get Stats
Check how many documents and chunks are indexed.

```bash
curl "http://localhost:8000/chat/stats"
```

**Response:**
```json
{
  "total_chunks_indexed": 45,
  "total_documents": 3,
  "embedding_dimension": 384,
  "next_vector_id": 45
}
```

### 5. Index Existing Document
Manually index a pre-uploaded document.

```bash
curl -X POST "http://localhost:8000/chat/index_document?doc_id=uuid-here"
```

## How FAISS Works

1. **Document Upload** → Split into chunks (800 chars, 120 overlap)
2. **Embeddings** → Convert to vectors using `all-MiniLM-L6-v2` (384 dims)
3. **FAISS Index** → Store vectors with L2 distance
4. **Retrieval** → Search for top-3 most similar chunks
5. **RAG** → Add context to LLM prompt

## File Structure

```
data/
├── uploads/           # Uploaded documents
│   └── {doc_id}.txt
├── faiss.index        # FAISS index (binary)
└── chunks.json        # Metadata (doc_id, source, text)
```

## Example Flow

```bash
# 1. Start server
python main.py

# 2. Upload a document
curl -X POST "http://localhost:8000/chat/upload_document" \
  -F "conversation_id=user1" \
  -F "file=@notes.txt"

# 3. Chat - automatically uses RAG
curl -X POST "http://localhost:8000/chat/user1" \
  -H "Content-Type: application/json" \
  -d '{"messages": "Summarize the document"}'

# 4. Check stats
curl "http://localhost:8000/chat/stats"
```

## Customization

Edit `llm.py` to adjust:
- `chunk_size`: How big chunks are (default: 800)
- `overlap`: Chunk overlap (default: 120)
- `k`: Number of chunks to retrieve (default: 4)
- Embedding model in `load_chat_model()`
