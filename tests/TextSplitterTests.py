import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RecursiveCharTextSplitter import RecursiveCharTextSplitter
from src.FaissVectorStore import FaissVectorStore

# ------------------------------------------------------------------------------

class TextSplitterTests(unittest.TestCase):
    def test_faiss_document_embedding(self):
        # Just a few paragrahps about langchain generated using an LLM.
        doc_file_paths = [
            '../../dataset/about-langchain.txt'
        ]
        vec_store = FaissVectorStore()
        vec_store.embed_documents(doc_file_paths)
        results = vec_store.search_similar(
            'LangChain provides abstractions to make working with LLMs easy',
            num_results=5
        )
        self.assertGreaterEqual(len(results), 1)
