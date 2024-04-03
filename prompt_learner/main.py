from adapters.openai import OpenAI
from adapters.anthropic import Anthropic

from templates.openai_template import OpenAICompletionTemplate
from templates.anthropic_template import AnthropicCompletionTemplate

from tasks.classification import ClassificationTask
from tasks.tagging import TaggingTask

from examples.example import Example

from optimizers.selectors.random_sampler import RandomSampler
from prompts.cot import CoT
# Load environment variables from .env file
# openai = OpenAI().llm
# print(openai.invoke("who built you?"))
# anthropic = Anthropic().llm
# print(anthropic.invoke("who built you?"))

# classification_description="Classify images"
# classification_labels = ["Cat", "Dog", "Bird"]

# classification_task = ClassificationTask(description=classification_description,allowed_labels=classification_labels)
# classification_task.add_example(Example(text="A cat", label="Cat"))
# print(classification_task.examples)
# tagging_description="Tag parts of speech"
# tagging_allowed_labels = ["NOUN", "VERB", "ADJ"]
# tagging_task = TaggingTask(description=tagging_description, allowed_labels=tagging_allowed_labels)
# tagging_task.add_example(Example(text="The cat sat on the mat", label="NOUN,VERB,ADJ,NOUN"))
# print(tagging_task.examples)

# openai_template = OpenAICompletionTemplate(classification_task)
# print(openai_template.prompt)
# anthropic_template = AnthropicCompletionTemplate(classification_task)
#print(anthropic_template.prompt)


# openai_template = OpenAICompletionTemplate(tagging_task)
# print(openai_template.prompt)
# anthropic_template = AnthropicCompletionTemplate(tagging_task)
# print(anthropic_template.prompt)
# final_prompt = anthropic_template.add_prediction_sample("A cat")
# answer= classification_task.predict(Anthropic(), final_prompt)
# print("ANSWER:", answer)
# print("Valid output>", classification_task.validate_prediction("A cat", answer))
# final_prompt = openai_template.add_prediction_sample("A cat")
# answer = classification_task.predict(OpenAI(), final_prompt)
# print("ANSWER:", answer)
# print("Valid output>", classification_task.validate_prediction("A cat", answer))

classification_description = "You have to classify customer texts as Urgent or Not Urgent"
classification_labels = ["Urgent", "Not Urgent"]
classification_task = ClassificationTask(description=classification_description, allowed_labels=classification_labels)
classification_task.add_example(Example(text="I need help", label="Urgent"))
classification_task.add_example(Example(text="I got my package", label="Not Urgent"))
print(classification_task.examples)
task = classification_task
openai_template = OpenAICompletionTemplate(task=classification_task)
sampler = RandomSampler(num_samples=2, task=classification_task)
sampler.select_examples()
openai_prompt = CoT(template=openai_template, selector=sampler)
openai_prompt.assemble_prompt()

openai_prompt.add_inference("My package is missing")
print(openai_prompt.prompt)
answer = classification_task.predict(OpenAI(), openai_prompt.prompt)
print("ANSWER:", answer)
# anthropic_template = AnthropicCompletionTemplate(classification_task,sampler)

# print(anthropic_template.prompt)
# final_prompt = openai_template.add_prediction_sample("My package is missing")
# answer = classification_task.predict(OpenAI(), final_prompt)
# print("ANSWER:", answer)
# print("Valid output>", classification_task.validate_prediction("My package is missing", answer))
# final_prompt = anthropic_template.add_prediction_sample("My package is missing")
# answer = classification_task.predict(Anthropic(), final_prompt)
# print("ANSWER:", answer)
# print("Valid output>", classification_task.validate_prediction("My package is missing", answer))
