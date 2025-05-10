from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.TextSplitter import TextSplitter

class RecursiveCharTextSplitter(TextSplitter):
    def split_text(self, text_to_split):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = text_splitter.create_documents([text_to_split])
        return chunks