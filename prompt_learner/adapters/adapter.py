"""Class for a Generic Adapter"""
import re


class Adapter:
    """Defines the contract for a Generic Adapter."""
    def __init__(self, temperature: float = 1.0, max_tokens: int = 1024):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def process_output(self, output: str):
        """Process the output from the language model."""
        content = output.content.strip()
        content = content.replace("'", "")
        content = content.replace("`", "")
        content = content.replace("'", "")
        #if xml tag is not present, it will return the content as it is
        content = self.extract_xml_tag(content, "label")
        return content

    def extract_xml_tag(self, data: str, tag: str) -> str:
        """Extracts the data between the XML tags."""
        open_tag = "<" + tag + ">"
        close_tag = "</" + tag + ">"
        #replace open tag and close tag with empty string
        data = re.sub(open_tag, "", data)
        data = re.sub(close_tag, "", data)
        #replace new line characters
        data = re.sub(r"^\\n|\\n$", "", data)
        return data

    def __repr__(self):
        return "Generic Adapter"
