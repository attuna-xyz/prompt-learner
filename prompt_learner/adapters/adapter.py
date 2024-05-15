"""Class for a Generic Adapter"""


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
        return content