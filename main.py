from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from schema import UploadDocumentResponse
from schema import ChatRequest, ChatResponse
from database import ConversationManager
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid
from llm import Startup
from fastapi import Form, File, UploadFile
import os

app = FastAPI()
conversation_manager = ConversationManager()
startup = Startup()

@app.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(conversation_id: str, request: ChatRequest):

    llm = startup.chat_model

    # store user message
    await conversation_manager.add_message(conversation_id, "user", request.messages)

    # get conversation history
    prev_conversation = await conversation_manager.get_conversation(conversation_id)

    # convert stored conversation to LangChain messages
    messages = [SystemMessage(content="You are a helpful assistant.")]

    for msg in prev_conversation:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # add current user question
    messages.append(HumanMessage(content=request.messages))

    # call model
    response = await llm.ainvoke(messages)

    # store assistant response
    await conversation_manager.add_message(conversation_id, "assistant", response.content)

    return ChatResponse(response=response.content)

@app.post("/chat/upload_document", response_model=UploadDocumentResponse)
async def upload_document(conversation_id: str = Form(...), file: UploadFile = File(...)):
    UPLOAD_PATH = "data/uploads/"
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    doc_id = str(uuid.uuid4())

    try:
        # Read file content once
        content = await file.read()
        text_content = content.decode("utf-8")
        
        # Save to disk
        filepath = os.path.join(UPLOAD_PATH, f"{doc_id}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        # Add to FAISS index
        chunks_added = startup.add_document_to_index(text_content, doc_id, source=file.filename)
        
        # Log in conversation
        await conversation_manager.add_message(
            conversation_id, 
            "system", 
            f"Uploaded document: {file.filename} ({chunks_added} chunks indexed)"
        )
        
        return UploadDocumentResponse(
            document_id=doc_id,
            filename=file.filename,
            path=filepath,
            status="indexed"
        )
    except Exception as e:
        return UploadDocumentResponse(
            document_id=doc_id,
            filename=file.filename,
            path="",
            status=f"error: {str(e)}"
        )

@app.post("/chat/index_document")
def index_document(doc_id: str):
    try:
        with open(f"data/uploads/{doc_id}.txt", "r") as f:
            content = f.read()

            # function to create vector store for the document
            startup.add_document_to_index(content, doc_id)
    except FileNotFoundError:
        return {"error": "Document not found"}

@app.get("/chat/query")
def query_chat(conversation_id: str, user_message: str):
    # function to query vector store and get relevant chunks
    relevant_chunks = startup.query_index(user_message)

    # create a prompt for the LLM that includes the user message and relevant chunks
    prompt = f"User message: {user_message}\nRelevant information:\n"
    for chunk in relevant_chunks:
        prompt += f"- {chunk['text']}\n"

    # call the chat model with the prompt
    response = startup.chat_model.invoke([SystemMessage(content=prompt)])

    return {"response": response.content}