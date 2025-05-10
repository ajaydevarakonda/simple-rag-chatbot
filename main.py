import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.FaissVectorStore import FaissVectorStore
from src.DeepSeekOllamaConverserSimple import DeepSeekOllamaConverserSimple

# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[ + ] Initializing model...')
    # Initial docs to add to vector store.
    file_path_1 = os.path.join(os.getcwd(), 'dataset/about-langchain.txt')
    file_path_2 = os.path.join(os.getcwd(), 'dataset/about-faiss.txt')
    doc_file_paths = [file_path_1, file_path_2]
    faiss_vec_store = FaissVectorStore()
    faiss_vec_store.embed_documents(doc_file_paths)
    global converser
    converser = DeepSeekOllamaConverserSimple(faiss_vec_store.vector_store)
    print('[ + ] Done Initializing model!')
    yield

app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------------------------------

@app.get('/')
async def root(question: str):
    return await converser.query(question)
