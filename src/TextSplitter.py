from abc import ABC, abstractclassmethod

class TextSplitter(ABC):
    def split_text(self) -> list:
        pass