# Simple RAG Chatbot

# LangChain + Hugging Face + FAISS + Ollama: Minimal Integration Example
This repository is a simple, minimal prototype demonstrating how to integrate key components of an LLM-based application using:

- **LangChain** for orchestration
- **Hugging Face** models
- **FAISS** for local vector search
- **Ollama** for running language models locally
- **FastAPI and Uvicorn** to serve requests, allowing the server, exposing minimal endpoints to interact with the LangChain pipeline.

## Goals
- Provide a clear example of connecting modern language model tooling.
- Use LangChain as a unifying layer across model, memory, and retrieval components.
- Offer a minimal base for experimentation or extension.
- Planned Improvement: Implement per-user conversation history management using the provided user ID. Currently, the system maintains a unified history for all users, which is not ideal for personalized interactions.

## Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com/) installed and running

### Installation
```bash
pip install -r requirements.txt
```

## Run the Project
* Create a new directory `dataset/` in the project's and add some `.txt` files in it.

```bash
uvicorn main:app --port 8000
# In a new terminal / tab.
curl http://localhost:8000?question=Some%20question%20about%20this%20text
```

## Notes
* The project aims to stay close to the fundamentals.
* It is not production-ready, and intentionally avoids external UI or persistence layers.
* Designed to be easy to audit, modify, or extend.
