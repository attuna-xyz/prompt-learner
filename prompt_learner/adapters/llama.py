"""This module contains the Llama class
which is used to interact with the Fireworks API."""

import os
from langchain_fireworks import ChatFireworks
from dotenv import load_dotenv
from .adapter import Adapter


class Llama(Adapter):
    """An adapter for a llama model call using fireworks"""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1024, model_name: str = "llama-v3-70b-instruct"):
        super().__init__(temperature, max_tokens)
        load_dotenv()
        self.llm = ChatFireworks(
            model='accounts/fireworks/models/'+model_name,
            fireworks_api_key=os.getenv('FIREWORKS_API_KEY'),
            temperature=self.temperature,
            max_tokens=self.max_tokens)
