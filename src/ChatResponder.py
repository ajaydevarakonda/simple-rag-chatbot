from abc import ABC, abstractclassmethod
from dataclasses import dataclass

@dataclass
class ChatResponse:
    thought: str
    response: str

class ChatResponder(ABC):
    @abstractclassmethod
    def query(self, message) -> ChatResponse:
        pass