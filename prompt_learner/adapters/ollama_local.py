"""This module contains the Ollama class
which is used to interact with any model from Ollama."""

from langchain_community.llms import Ollama
from dotenv import load_dotenv
from .adapter import Adapter


class OllamaLocal(Adapter):
    """An adapter for calling a local model using ollama"""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1024, model_name: str = "llama3"):
        super().__init__(temperature, max_tokens)
        load_dotenv()
        self.llm = Ollama(
            model=model_name,
            temperature=self.temperature)

    def process_output(self, output: str):
        """Process the output from the language model."""
        content = output.strip()
        content = content.replace("'", "")
        content = content.replace("`", "")
        content = content.replace("'", "")
        #if xml tag is not present, it will return the content as it is
        content = self.extract_xml_tag(content, "label")
        return content
    
    def __repr__(self):
        return f"""Ollama Adapter(model_name={self.llm.model})"""