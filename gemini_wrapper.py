# gemini_wrapper.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage, ChatResult, ChatGeneration

# Load environment variables from .env file
load_dotenv()

class GeminiEmbeddings(Embeddings):
    """Embeddings using Gemini API"""
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")
        
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-embedding-001"

    def embed_documents(self, texts):
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="retrieval_document")
        )
        return [list(e.values) if hasattr(e, "values") else e for e in response.embeddings]

    def embed_query(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=types.EmbedContentConfig(task_type="retrieval_query")
        )
        e = response.embeddings[0]
        return list(e.values) if hasattr(e, "values") else e


class GeminiChat(BaseChatModel):
    """Gemini chat model for RAG"""
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": self.temperature}

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")
        
        client = genai.Client(api_key=api_key)
        
        text_messages = [msg.content for msg in messages]
        response = client.models.generate_content(
            model=self.model_name,
            contents=text_messages,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        gen = ChatGeneration(message=HumanMessage(content=response.text))
        return ChatResult(generations=[gen])
