from abc import ABC, abstractclassmethod

class ContentGetter(ABC):
    @abstractclassmethod
    def get_content(self) -> str:
        pass