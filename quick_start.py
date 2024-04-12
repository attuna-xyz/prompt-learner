from prompt_learner.tasks.classification import ClassificationTask
from prompt_learner.examples.example import Example
from prompt_learner.prompts.prompt import Prompt
from prompt_learner.templates.openai_template import OpenAICompletionTemplate
from prompt_learner.adapters.openai import OpenAI
from prompt_learner.optimizers.selectors.random_sampler import RandomSampler

classification_task = ClassificationTask(description="You have to classify customer texts as Urgent or Not Urgent",
                                        allowed_labels=["Urgent", "Not Urgent"])

classification_task.add_example(Example(text="I need help", label="Urgent"))
classification_task.add_example(Example(text="I got my package", label="Not Urgent"))


template = OpenAICompletionTemplate(task=classification_task) #rename to Gpt template

openai_prompt = Prompt(template=template, selector= RandomSampler(num_samples=1, task=classification_task).select_examples())
openai_prompt.assemble_prompt()

print(openai_prompt.prompt)

openai_prompt.add_inference("My package is missing")
answer = classification_task.predict(OpenAI(), openai_prompt.prompt)
print(answer)