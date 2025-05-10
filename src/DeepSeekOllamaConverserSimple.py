from typing import TypedDict, Annotated, List

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM

from src.ChatResponder import ChatResponder, ChatResponse

# ------------------------------------------------------------------------------

class ChatResponse:
    def __init__(self, thought, response):
        self.thought = thought
        self.response = response

class DeepSeekOllamaConverserSimple(ChatResponder):
    def __init__(self, vector_store, model_name='deepseek-r1:14b'):
        self.vector_store = vector_store
        self.llm = OllamaLLM(model=model_name)
        self.retriever_prompt = ChatPromptTemplate.from_messages([
            ('system', 'You are a helpful AI assistant. Use the following pieces of retrieved context to answer the user\'s question. If you don\'t know the answer, just say you don\'t know — don\'t try to make up an answer.'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', 'Context:\n{context}\n\n, Question: {input}')
        ])
        self.init_retrieval_chain()
        self.chat_history: List[BaseMessage] = []

    def init_retrieval_chain(self):
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, self.retriever_prompt
        )
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.as_retriever(),
            combine_docs_chain=combine_docs_chain
        )

    async def query(self, question: str) -> ChatResponse:
        result = await self.retrieval_chain.ainvoke({
            'input': question,
            'chat_history': self.chat_history,
        })

        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=result['answer'])
        ])

        answer = result['answer']
        thought, response = answer.split('</think>')
        thought = thought.replace('<think>', '').strip()
        response = response.strip()
        return ChatResponse(thought, response)
