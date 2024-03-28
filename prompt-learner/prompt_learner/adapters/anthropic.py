from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

class Anthropic:
    def __init__(self):
        load_dotenv()
        self.llm = ChatAnthropic(anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),model="claude-3-haiku-20240307", temperature=0.0, max_tokens=1024)
