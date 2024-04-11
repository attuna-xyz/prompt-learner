"""This module contains the Anthropic class,
which is an adapter for the Anthropic language model API."""

import os
import re
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from .adapter import Adapter


class Anthropic(Adapter):
    """An adapter for an Anthropic language model call"""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1024):
        super().__init__(temperature, max_tokens)
        load_dotenv()
        self.llm = ChatAnthropic(
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                model="claude-3-haiku-20240307",
                temperature=self.temperature,
                max_tokens=self.max_tokens)

    def extract_xml_tag(self, data: str, tag: str) -> str:
        """Extracts the data between the XML tags."""
        open_tag = "<" + tag + ">"
        close_tag = "</" + tag + ">"
        try:
            data = (data.split(open_tag)[1]).split(close_tag)[0].strip()
        except IndexError:
            pass
        data = re.sub(r"^\\n|\\n$", "", data)
        return data
    
    def process_output(self, output: str):
        """Process the output from the language model."""
        return self.extract_xml_tag(output.content.strip(), "label")
