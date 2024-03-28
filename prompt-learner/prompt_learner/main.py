from adapters.openai import OpenAI
from adapters.anthropic import Anthropic
  # Load environment variables from .env file
openai = OpenAI().llm
print(openai.invoke("who built you?"))
anthropic = Anthropic().llm
print(anthropic.invoke("who built you?"))
