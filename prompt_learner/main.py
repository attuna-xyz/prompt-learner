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

classification_description="Classify images"
classification_labels = ["Cat", "Dog", "Bird"]

classification_task = ClassificationTask(description=classification_description,allowed_labels=classification_labels)
classification_task.add_example(Example(text="A cat", label="Cat"))
print(classification_task.examples)
tagging_description="Tag parts of speech"
tagging_allowed_labels = ["NOUN", "VERB", "ADJ"]
tagging_task = TaggingTask(description=tagging_description, allowed_labels=tagging_allowed_labels)
tagging_task.add_example(Example(text="The cat sat on the mat", label="NOUN,VERB,ADJ,NOUN"))
print(tagging_task.examples)

openai_template = OpenAICompletionTemplate(classification_task)
print(openai_template.prompt)
anthropic_template = AnthropicCompletionTemplate(classification_task)
print(anthropic_template.prompt)


# openai_template = OpenAICompletionTemplate(tagging_task)
# print(openai_template.prompt)
# anthropic_template = AnthropicCompletionTemplate(tagging_task)
# print(anthropic_template.prompt)
final_prompt = anthropic_template.add_prediction_sample("A cat")
print("ANSWER:", classification_task.predict(Anthropic(), final_prompt))