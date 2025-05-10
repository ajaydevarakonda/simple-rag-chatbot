from uuid import uuid4

from src.TextSplitter import TextSplitter
from src.ContentGetter import ContentGetter

class MultiFileVectorStoreEmbedder:
    def __init__(
        self, content_getter: ContentGetter, text_splitter: TextSplitter,
        vector_store,
    ):
        self.content_getter = content_getter
        self.text_splitter = text_splitter
        self.vector_store = vector_store

    def embedd_docs_to_vector_store(self, file_paths):
        content = self.content_getter.get_content(file_paths)
        documents = self.text_splitter.split_text(content)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        return self.vector_store
