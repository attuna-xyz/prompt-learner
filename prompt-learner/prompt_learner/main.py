from adapters.openai import OpenAI
from adapters.anthropic import Anthropic

from templates.openai_template import OpenAICompletionTemplate
from templates.anthropic_template import AnthropicCompletionTemplate
  # Load environment variables from .env file
# openai = OpenAI().llm
# print(openai.invoke("who built you?"))
# anthropic = Anthropic().llm
# print(anthropic.invoke("who built you?"))
task_type="classification"
task_description="classify the given text as urgent or not urgent"
openai_template = OpenAICompletionTemplate(task_description,task_type)
print(openai_template.template)
anthropic_template = AnthropicCompletionTemplate(task_description,task_type)
print(anthropic_template.template)

