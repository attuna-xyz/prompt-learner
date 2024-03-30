from adapters.openai import OpenAI
from adapters.anthropic import Anthropic

from templates.openai_template import OpenAICompletionTemplate
from templates.anthropic_template import AnthropicCompletionTemplate

from tasks.classification import ClassificationTask
from tasks.tagging import TaggingTask

from examples.example import Example
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

classification_task = ClassificationTask(name="Image Classification", allowed_labels=["Cat", "Dog", "Bird"])
classification_task.add_example(Example(text="A cat", label="Cat"))
print(classification_task.examples)
tagging_task = TaggingTask(name="Part of Speech Tagging", allowed_labels=["NOUN", "VERB", "ADJ"])
tagging_task.add_example(Example(text="The cat sat on the mat", label="NOUN,VERB,ADJ,NOUN"))
print(tagging_task.examples)