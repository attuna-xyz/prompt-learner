from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

class OpenAI:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(openai_api_key = os.getenv('OPENAI_API_KEY'),temperature=0.0, max_tokens=1024)