"""This module contains the Anthropic class,
which is an adapter for the Anthropic language model API."""

import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv


class Anthropic:
    """An adapter for an Anthropic language model call"""
    def __init__(self):
        load_dotenv()
        self.llm = ChatAnthropic(
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                model="claude-3-haiku-20240307",
                temperature=0.0,
                max_tokens=1024)
