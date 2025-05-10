import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.MultiFileContentGetter import MultiFileContentGetter
from src.RecursiveCharTextSplitter import RecursiveCharTextSplitter
from src.MultiFileVectorStoreEmbedder import MultiFileVectorStoreEmbedder

# ------------------------------------------------------------------------------

class FaissVectorStore:
    def __init__(self):
        self.content_getter = MultiFileContentGetter()
        self.text_splitter = RecursiveCharTextSplitter()
        embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        index = faiss.IndexFlatL2(len(embedder.embed_query('hello world')))
        self.vector_store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.mf_vs_embedder = MultiFileVectorStoreEmbedder(
            self.content_getter,
            self.text_splitter,
            self.vector_store,
        )

    def embed_documents(self, file_paths=[]):
        self.vector_store = self.mf_vs_embedder.embedd_docs_to_vector_store(
            file_paths
        )
        
    def search_similar(self, search_str, filter={}, num_results=2):
        results = self.vector_store.similarity_search(
            search_str,
            k=num_results,
            filter=filter,
        )
        return results
