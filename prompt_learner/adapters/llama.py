"""This module contains the Llama class
which is used to interact with the Fireworks API."""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from .adapter import Adapter


class Llama(Adapter):
    """An adapter for a llama model call using fireworks"""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1024, model_name: str = "llama3-8b-8192"):
        super().__init__(temperature, max_tokens)
        load_dotenv()
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=os.getenv('GROQ_API_KEY'),
            temperature=self.temperature,
            max_tokens=self.max_tokens)
    
    def __repr__(self):
        return f"""Llama Adapter(model_name={self.llm.model_name})"""