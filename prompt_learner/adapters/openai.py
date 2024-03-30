"""This module contains the OpenAI class
which is used to interact with the OpenAI API."""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class OpenAI:
    """An adapter for an OpenAI language model call"""
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.0,
            max_tokens=1024)
