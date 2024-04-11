"""This module contains the OpenAI class
which is used to interact with the OpenAI API."""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from .adapter import Adapter


class OpenAI(Adapter):
    """An adapter for an OpenAI language model call"""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1024):
        super().__init__(temperature, max_tokens)
        load_dotenv()
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            temperature=self.temperature,
            max_tokens=self.max_tokens)
